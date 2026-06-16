pip install -r requirements_opd.txt

pip install flash-attn==2.8.1 --no-build-isolation
pip install -e . --no-build-isolation --no-deps
pip install nemo-automodel==0.2.0