@echo off
echo Installing/Verifying Dependencies...
pip install flask flask-cors google-generativeai sounddevice numpy scipy python-dotenv

echo Starting Balti Voice AI (Flask Web Server)...
cd "Balti AI"
start http://127.0.0.1:5000
python main.py
pause
