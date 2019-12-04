mkdir -p .log

python local/normalize.py --output_path=ATLAS  2>&1 | tee .log/normalize.log 
python local/normalize.py --output_path=MR  2>&1 | tee .log/normalize.log 
python local/normalize.py --output_path=CT  2>&1 | tee .log/normalize.log 