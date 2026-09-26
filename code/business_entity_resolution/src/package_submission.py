"""Build the required archive only after both output files validate."""
import argparse
from pathlib import Path
import re
import zipfile
from validate_submission import validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--team-name',required=True)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--test-dir',type=Path,default=Path('student_resource/dataset/test'))
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.team_name):
        p.error('Team name must use letters, digits, underscores or hyphens')
    root=args.root
    outputs=[root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv']
    doc=root/'Documentation_template.md'
    for path in outputs+[doc]:
        if not path.is_file():
            p.error(f'Missing required deliverable: {path}')
    errors,warnings=validate(str(outputs[0]),str(outputs[1]),str(args.test_dir),check_ids=True)
    if errors or warnings:
        raise SystemExit('\n'.join(errors+warnings))
    code=root/'code/business_entity_resolution'
    files=outputs+[doc,code/'README.md',code/'requirements.txt',code/'MODEL_LICENSE']+sorted((code/'src').glob('*.py'))
    archive=root/f'{args.team_name}_submission.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            z.write(file,file.relative_to(root))
    print(archive)

if __name__=='__main__':
    main()
