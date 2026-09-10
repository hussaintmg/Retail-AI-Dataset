from pathlib import Path
import json, shutil, subprocess

SRC = Path('_sources/grocery-store-shelves')
DST = Path('datasets/retail_detection/supervisely_shelves_real')

if SRC.exists():
    shutil.rmtree(SRC)
SRC.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(['git','clone','--depth','1','https://github.com/supervisely-ecosystem/grocery-store-shelves.git',str(SRC)], check=True)

if DST.exists():
    shutil.rmtree(DST)
for split in ('train','val','test'):
    (DST/'images'/split).mkdir(parents=True, exist_ok=True)
    (DST/'labels'/split).mkdir(parents=True, exist_ok=True)

img_dir = SRC/'project'/'ds0'/'img'
ann_dir = SRC/'project'/'ds0'/'ann'
images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])

def split_for(i, n):
    r = i / max(n,1)
    return 'train' if r < .70 else ('val' if r < .85 else 'test')

box_total = 0
for i, img in enumerate(images):
    ann_path = ann_dir / f'{img.name}.json'
    if not ann_path.exists():
        continue
    ann = json.loads(ann_path.read_text(encoding='utf-8'))
    w = float(ann['size']['width']); h = float(ann['size']['height'])
    rows=[]
    for obj in ann.get('objects',[]):
        if obj.get('classTitle') != 'product' or obj.get('geometryType') != 'rectangle':
            continue
        (x1,y1),(x2,y2)=obj['points']['exterior']
        x1=max(0,min(float(x1),w)); x2=max(0,min(float(x2),w))
        y1=max(0,min(float(y1),h)); y2=max(0,min(float(y2),h))
        if x2 <= x1 or y2 <= y1:
            continue
        xc=((x1+x2)/2)/w; yc=((y1+y2)/2)/h; bw=(x2-x1)/w; bh=(y2-y1)/h
        rows.append(f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')
    if not rows:
        continue
    split=split_for(i,len(images))
    shutil.copy2(img, DST/'images'/split/img.name)
    (DST/'labels'/split/f'{img.stem}.txt').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    box_total += len(rows)

(DST/'data.yaml').write_text('path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: product\n',encoding='utf-8')

actual_imgs=sum(1 for p in (DST/'images').rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'})
actual_labels=sum(1 for p in (DST/'labels').rglob('*.txt'))
print(f'Imported {actual_imgs} real shelf images, {actual_labels} label files, {box_total} product boxes')
assert actual_imgs >= 30
assert actual_imgs == actual_labels
assert box_total >= 1000
