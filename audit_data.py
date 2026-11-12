"""Read-only streaming audit. Titles are noisy proxies, never ground truth.

No CSV escapechar is set: RFC-style doubled quotes and literal backslashes
must be preserved. This script fails on malformed rows rather than skipping.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

STARS = {"one star": 1, "two stars": 2, "three stars": 3,
         "four stars": 4, "five stars": 5}

def norm(s):
    return " ".join(s.lower().split())

def run(args):
    started = time.perf_counter()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(args.test, newline="", encoding="utf-8-sig") as f:
        test = list(csv.DictReader(f))
    text_keys = {(norm(r['title']), norm(r['comment'])) for r in test}
    pair_keys = {(r['user_id'], r['prod_id']) for r in test}
    identity_keys = {(r['user_id'], r['prod_id'], r['time'], norm(r['title']), norm(r['comment'])) for r in test}
    text_hist = defaultdict(Counter); pair_hist = defaultdict(Counter)
    identity_hist = defaultdict(Counter)
    confusion = [[0]*5 for _ in range(5)]
    by_block = defaultdict(lambda: [[0]*5 for _ in range(5)])
    labels = Counter(); bad = Counter(); n = 0
    sample_file = open(out/'pilot_sample.csv', 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(sample_file, fieldnames=['id','title','comment','rating'])
    writer.writeheader()
    sampled = 0
    with open(args.train, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, strict=True)
        for r in reader:
            if None in r or any(v is None for v in r.values()):
                raise ValueError(f"Malformed CSV at parsed record {n}")
            y = int(r['rating']); assert 1 <= y <= 5
            n += 1; labels[y] += 1
            title, comment = norm(r['title']), norm(r['comment'])
            s = STARS.get(title, 0)
            if s:
                confusion[s-1][y-1] += 1
                by_block[(n-1)//1000000][s-1][y-1] += 1
            if not comment: bad['empty_comment'] += 1
            if int(r['votes']) < 0: bad['negative_votes'] += 1
            k = (title, comment)
            if k in text_keys: text_hist[k][y] += 1
            k2 = (r['user_id'], r['prod_id'])
            if k2 in pair_keys: pair_hist[k2][y] += 1
            ident = (r['user_id'], r['prod_id'], r['time'], title, comment)
            if ident in identity_keys: identity_hist[ident][y] += 1
            # Hash sampling spreads the pilot over the full file, not its prefix.
            if int.from_bytes(hashlib.blake2b(r['id'].encode(),digest_size=8).digest(),'little') % args.sample_every == 0:
                writer.writerow({k:r[k] for k in writer.fieldnames}); sampled += 1
            if n % 1000000 == 0: print(f"parsed {n:,} records", flush=True)
            if args.limit and n >= args.limit: break
    sample_file.close()
    total_star = sum(map(sum,confusion))
    agreement = sum(confusion[i][i] for i in range(5))/total_star
    transition = [[v/sum(row) if sum(row) else 0 for v in row] for row in confusion]
    row_means = [sum((j+1)*p for j,p in enumerate(row)) for row in transition]
    # Symmetric noise MODEL: T = a I + (1-a) J/5. Not an established mechanism.
    a = (sum(transition[i][i] for i in range(5))/5 - .2)/.8
    prior_observed = [labels[i]/n for i in range(1,6)]
    prior_latent = [(p-(1-a)/5)/a for p in prior_observed]
    conditional_variance = [a*z*z+(1-a)*11-(a*z+(1-a)*3)**2 for z in range(1,6)]
    noise_floor = sum(p*v for p,v in zip(prior_latent,conditional_variance))
    def retrieval_summary(hist, keyfn):
        matched = [r for r in test if keyfn(r) in hist]
        conflict = [r for r in matched if len(hist[keyfn(r)]) > 1]
        examples = []
        for k,c in sorted(hist.items(),key=lambda kv:sum(kv[1].values()),reverse=True)[:5]:
            num=sum(c.values()); mu=sum(y*v for y,v in c.items())/num
            var=sum((y-mu)**2*v for y,v in c.items())/num
            examples.append({'key':k,'n':num,'label_counts':dict(c),'mean':mu,'std':math.sqrt(var)})
        return {'test_coverage_rows':len(matched),'test_conflicting_match_rows':len(conflict),'examples':examples}
    result = dict(rows=n,sampled=sampled,rating_counts=dict(labels),quality=dict(bad),
                  title_proxy_confusion=confusion,title_proxy_transition=transition,
                  title_proxy_agreement=agreement,title_proxy_conditional_means=row_means,
                  title_proxy_by_million=by_block,
                  symmetric_noise_hypothesis={'a':a,'uniform_replacement_probability':1-a,
                    'latent_prior_estimate':prior_latent,'conditional_noise_floor_rmse':math.sqrt(noise_floor) if noise_floor>=0 else None,
                    'warning':'Conditional on latent-title validity, independent symmetric corruption and identical test noise; NOT an unconditional achievable score bound.'},
                  normalized_text=retrieval_summary(text_hist,lambda r:(norm(r['title']),norm(r['comment']))),
                  user_product=retrieval_summary(pair_hist,lambda r:(r['user_id'],r['prod_id'])),
                  review_identity=retrieval_summary(identity_hist,lambda r:(r['user_id'],r['prod_id'],r['time'],norm(r['title']),norm(r['comment']))),
                  seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['title_proxy_by_million']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--test',required=True)
    p.add_argument('--out',required=True); p.add_argument('--sample-every',type=int,default=32); p.add_argument('--limit',type=int)
    a=p.parse_args(); assert a.sample_every>0
    csv.field_size_limit(10000000)
    run(a)
