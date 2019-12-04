mkdir -p .log

python local/skullstrip.py --output_path=ATLAS  2>&1 | tee .log/skullstrip.log 
python local/skullstrip.py --output_path=MR  2>&1 | tee .log/skullstrip.log 
python local/skullstrip.py --output_path=CT  2>&1 | tee .log/skullstrip.log 