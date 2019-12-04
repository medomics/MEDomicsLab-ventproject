mkdir -p .log

python local/dicom2tensor.py --input_path=ATLAS_PATH --output_path=ATLAS  2>&1 | tee .log/dicom2tensor.log 
python local/dicom2tensor.py --input_path=MR_PATH --output_path=MR 2>&1 | tee .log/dicom2tensor.log
python local/dicom2tensor.py --input_path=CT_PATH --output_path=CT 2>&1 | tee .log/dicom2tensor.log
python local/dicom2tensor.py --input_path=MR_TEST2_PATH --output_path=MR 2>&1 | tee .log/dicom2tensor.log
python local/dicom2tensor.py --input_path=CT_TEST2_PATH --output_path=CT 2>&1 | tee .log/dicom2tensor.log

