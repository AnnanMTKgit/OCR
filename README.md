# Pour telecharger le projet 
git clone https://github.com/AnnanMTKgit/OCR.git

# Install dependences 
pip install -r requirements.txt

# Launch app 
uvicorn api:app --reload

To test, upload recto and verso bytes files contained in folder testbytesImages //
For exemple in the folder testbytesImages adji.bytes is recto file and adji_v.bytes is verso file
