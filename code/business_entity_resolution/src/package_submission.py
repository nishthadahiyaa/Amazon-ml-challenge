#!/usr/bin/env python3
"""Build the required submission archive only after both output files exist."""
import argparse
from pathlib import Path
import re
import sys
import zipfile
from validate_submission import validate


def main():
    p = argparse.ArgumentParser(description="Package submission into <team_name>_submission.zip")
    p.add_argument('--team-name', required=True, help="Team name for submission zip")
    p.add_argument('--root', type=Path, default=Path('.'), help="Root directory")
    p.add_argument('--test-dir', type=Path, default=Path('student_resource/dataset/test'), help="Path to test dataset")
    p.add_argument('--skip-validation', action='store_true', help="Skip validator checks before packaging")
    args = p.parse_args()

    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.team_name):
        p.error('Team name must use letters, digits, underscores or hyphens')

    root = args.root
    outputs = [root / 'output/matching_results.tsv', root / 'output/candidate_pairs.tsv']
    doc = root / 'Documentation_template.md'

    for path in outputs + [doc]:
        if not path.is_file():
            p.error(f'Missing required deliverable: {path}')

    if not args.skip_validation:
        errors, warnings = validate(str(outputs[0]), str(outputs[1]), str(args.test_dir), check_ids=False)
        if warnings:
            print("Validator warnings:\n" + "\n".join(warnings), file=sys.stderr)
        if errors:
            raise SystemExit("Validator errors:\n" + "\n".join(errors))

    code = root / 'code/business_entity_resolution'
    files = outputs + [doc, code / 'README.md', code / 'requirements.txt']
    if (code / 'MODEL_LICENSE').is_file():
        files.append(code / 'MODEL_LICENSE')
    files += sorted((code / 'src').glob('*.py'))

    archive = root / f'{args.team_name}_submission.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            z.write(file, file.relative_to(root))

    print(f"Created submission archive: {archive}")


if __name__ == '__main__':
    main()
