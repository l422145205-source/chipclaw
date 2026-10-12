"""Lightweight, model-free workbench and shared static UI installation."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]


def install_workbench(app: FastAPI, *, live: bool = False):
    @app.get('/workbench/status')
    def status():
        return {'live': live, 'mode': 'live' if live else 'recorded',
                'note': 'Recorded runs do not execute models. Live mode requires local model and corpus assets.'}

    app.mount('/assets', StaticFiles(directory=ROOT / 'web' / 'assets'), name='assets')
    app.mount('/demo', StaticFiles(directory=ROOT / 'web' / 'demo'), name='demo')


app = FastAPI(title='ChipClaw Workbench — recorded runs', version='1.0')
install_workbench(app)


@app.get('/')
def home():
    return FileResponse(ROOT / 'web' / 'index.html')


@app.get('/health')
def health():
    return {'ready': True, 'mode': 'recorded', 'generation_calls': 0}
