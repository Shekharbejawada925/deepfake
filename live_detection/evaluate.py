"""Evaluate checkpoint on a held-out folder of labelled face/video frame windows.

Manifest CSV columns: path,label,group  where path points to a directory of
exactly 8 JPEG/PNG face-containing frames, label is REAL or FAKE, and group is
a subject/source identity. This script never trains or infers truth labels.
"""
import argparse
import csv
import json
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest",type=Path)
    p.add_argument("--output",type=Path,default=Path("evaluation_results.json"))
    a=p.parse_args()
    from legacy_model_adapter import predict_frames
    rows=list(csv.DictReader(a.manifest.open(newline="")))
    results=[]
    for row in rows:
        label=row["label"].upper()
        if label not in ("REAL","FAKE"):
            raise ValueError("Invalid label; expected REAL or FAKE")
        folder=Path(row["path"])
        paths=sorted([x for x in folder.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png")])
        if len(paths)!=8:
            raise ValueError(f"Expected exactly 8 frames in {folder}")
        images=[]
        for f in paths:
            with Image.open(f) as im:
                images.append(im.convert("RGB").copy())
        try:
            prob=predict_frames(images)
            status="scored"
        except ValueError as exc:
            prob=None
            status=str(exc)
        results.append({"path":str(folder),"label":label,"group":row.get("group",""),
                        "fake_probability":prob,"status":status})
    scored=[r for r in results if r["fake_probability"] is not None]
    tp=sum(r["fake_probability"]>=.5 and r["label"]=="FAKE" for r in scored)
    fp=sum(r["fake_probability"]>=.5 and r["label"]=="REAL" for r in scored)
    tn=sum(r["fake_probability"]<.5 and r["label"]=="REAL" for r in scored)
    fn=sum(r["fake_probability"]<.5 and r["label"]=="FAKE" for r in scored)
    def div(n,d):
        return round(n/d,4) if d else None
    report={"total":len(results),"scored":len(scored),"abstained":len(results)-len(scored),
            "coverage":div(len(scored),len(results)), "confusion":{"TP":tp,"FP":fp,"TN":tn,"FN":fn},
            "accuracy_scored_only":div(tp+tn,len(scored)),
            "precision_fake":div(tp,tp+fp),"recall_fake":div(tp,tp+fn),
            "specificity_real":div(tn,tn+fp),
            "warning":"Metrics require independent ground-truth labels, source/identity-disjoint splits, and verified class mapping.",
            "results":results}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))
if __name__=="__main__":
    main()
