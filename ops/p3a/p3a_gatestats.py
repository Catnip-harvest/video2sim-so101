import json, collections, sys
rows=sorted([json.loads(l) for l in open(sys.argv[1])], key=lambda r:r['attempt'])
rows=[r for r in rows if r['attempt']<200]
print('n',len(rows),'acc',sum(r['accepted'] for r in rows))
print('primary', dict(collections.Counter(r['primary_reason'] for r in rows if not r['accepted'])))
print('any', dict(collections.Counter(x for r in rows if not r['accepted'] for x in r['reasons'])))
acc=[r for r in rows if r['accepted']]
print('accepted distractor max move mm: max',max(max(r['stats']['distractor_max_move_mm']) for r in acc),'pick max',max(max(r['stats']['distractor_max_move_pick_mm']) for r in acc))
allm=[max(r['stats'].get('distractor_max_move_mm',[0])) for r in rows if r['stats']]
print('all attempts n>=1mm', sum(x>=1 for x in allm), 'max', max(allm))
cont=[r['stats'].get('distractor_contacts') for r in rows if r['stats']]
print('arm contacts', sum(c['arm']>0 for c in cont), 'obj contacts', sum(c['object']>0 for c in cont))
print('both jaws all carry steps', all(r['stats']['carry_both_jaw_steps']==r['stats']['carry_steps']>0 for r in acc))
print(dict(collections.Counter(r['variation']['extras']['multitask']['target_colour'] for r in rows)), dict(collections.Counter(r['variation']['extras']['multitask']['target_colour'] for r in acc)))
