from pathlib import Path
import csv, shutil, subprocess

ROOT=Path('datasets/product_recognition/grocery_store_real')
SRC=Path('/tmp/GroceryStoreDataset')

if SRC.exists(): shutil.rmtree(SRC)
subprocess.run(['git','clone','--depth','1','https://github.com/marcusklasson/GroceryStoreDataset.git',str(SRC)],check=True)

if ROOT.exists(): shutil.rmtree(ROOT)
for split in ('train','val','test'):
    (ROOT/'images'/split).mkdir(parents=True,exist_ok=True)
    (ROOT/'labels'/split).mkdir(parents=True,exist_ok=True)

# class map from official classes.csv
classes=[]
with open(SRC/'dataset/classes.csv',newline='',encoding='utf-8') as f:
    r=csv.reader(f)
    header=next(r)
    for row in r:
        if not row: continue
        classes.append(row[0].strip())

# Fallback to numeric names if schema changes
if not classes:
    classes=[f'class_{i:03d}' for i in range(81)]

for split in ('train','val','test'):
    list_file=SRC/'dataset'/f'{split}.txt'
    with open(list_file,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=[p.strip() for p in line.split(',')]
            rel=parts[0]
            fine_id=int(parts[1])
            src=SRC/'dataset'/rel
            if not src.exists():
                raise FileNotFoundError(src)
            # stable unique filename preserves category path in stem
            stem=rel.replace('/','__').rsplit('.',1)[0]
            ext=src.suffix.lower()
            dst=ROOT/'images'/split/f'{stem}{ext}'
            shutil.copy2(src,dst)
            (ROOT/'labels'/split/f'{stem}.txt').write_text(f'{fine_id}\n',encoding='utf-8')

# only dataset metadata inside dataset folder
lines=['path: .','train: images/train','val: images/val','test: images/test','','names:']
for i,name in enumerate(classes):
    lines.append(f'  {i}: {name}')
(ROOT/'data.yaml').write_text('\n'.join(lines)+'\n',encoding='utf-8')

images=list(ROOT.rglob('*.jpg'))+list(ROOT.rglob('*.jpeg'))+list(ROOT.rglob('*.png'))
labels=list((ROOT/'labels').rglob('*.txt'))
print(f'Imported real images: {len(images)}')
print(f'Imported labels: {len(labels)}')
if len(images) != len(labels): raise SystemExit('image/label mismatch')
if len(images) < 5000: raise SystemExit('unexpectedly small import')
