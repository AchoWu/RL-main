pip install torch==2.9.0
pip install -r requirements_opd.txt

pip install flash-attn==2.8.1 --no-build-isolation
pip install -e . --no-build-isolation --no-deps
pip install nemo-automodel==0.2.0
pip uninstall torchao -y
