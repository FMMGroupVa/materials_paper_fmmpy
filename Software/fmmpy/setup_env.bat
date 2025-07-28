@echo off
REM Crear entorno virtual
python -m venv .venv

REM Activar entorno
call .venv\Scripts\activate

REM Actualizar pip y herramientas
pip install --upgrade pip
pip install build twine pytest

REM Instalar dependencias mínimas compatibles con fmmpy
pip install numpy>=1.25,<2.0 scipy>=1.10 numba>=0.57 pandas>=1.5 matplotlib>=3.7 qpsolvers>=3.0 quadprog>=0.1.13

REM Instalar fmmpy desde el paquete ya compilado
pip install dist\fmmpy-0.1.0-py3-none-any.whl

REM Confirmar instalación
echo.
echo =============================
echo ENVIRONMENT READY FOR FMM
echo To activate it, run:
echo call .venv\Scripts\activate
echo =============================