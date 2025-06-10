"""
jupyter_config.py - Configuración de Jupyter para Football Analytics

Configuración personalizada de Jupyter Lab/Notebook específicamente optimizada
para análisis de datos deportivos y desarrollo de modelos ML.

Author: Football Analytics Team
Version: 2.1.0
Date: 2024-06-02

Uso:
    jupyter lab --config=jupyter_config.py
    jupyter notebook --config=jupyter_config.py
"""

import os
import sys
from pathlib import Path

# ================================
# CONFIGURACIÓN BASE
# ================================

# Configuración del servidor Jupyter
c = get_config()

# Información del proyecto
PROJECT_NAME = "Football Analytics"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"

# ================================
# CONFIGURACIÓN DEL SERVIDOR
# ================================

# Configuración de red
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True

# Configuración de seguridad para desarrollo
# NOTA: En producción, usar tokens y passwords seguros
c.ServerApp.token = ""
c.ServerApp.password = ""
c.ServerApp.disable_check_xsrf = True

# Configuración de CORS para desarrollo
c.ServerApp.allow_origin = "*"
c.ServerApp.allow_credentials = True

# Directorio base de notebooks
c.ServerApp.root_dir = str(PROJECT_ROOT)
c.ServerApp.notebook_dir = str(NOTEBOOKS_DIR)

# ================================
# CONFIGURACIÓN DE JUPYTER LAB
# ================================

# Configuración por defecto de Jupyter Lab
c.LabApp.default_url = "/lab"
c.LabApp.app_dir = str(PROJECT_ROOT / ".jupyter" / "lab")

# Extensiones habilitadas por defecto
c.LabServerApp.collaborative = False

# ================================
# CONFIGURACIÓN DE CONTENIDO
# ================================

# Formatos de archivo soportados
c.ContentsManager.allowed_formats = [
    "json",
    "py",
    "ipynb",
    "md",
    "txt",
    "csv",
    "tsv",
    "html",
    "xml",
    "sql",
    "yaml",
    "yml",
]

# Tamaño máximo de archivos (50MB para datasets)
c.ContentsManager.max_file_size_to_edit = 52428800

# ================================
# CONFIGURACIÓN DEL KERNEL
# ================================

# Configuración del kernel Python
c.KernelManager.default_kernel_name = "python3"

# Timeout para kernels (útil para entrenamientos largos de ML)
c.KernelManager.kernel_info_timeout = 60
c.MappingKernelManager.default_kernel_name = "python3"

# Configuración de recursos para kernels
c.KernelManager.shutdown_wait_time = 5.0

# ================================
# CONFIGURACIÓN ESPECÍFICA DE FOOTBALL ANALYTICS
# ================================

# Variables de entorno para notebooks
c.Spawner.environment = {
    "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT}/app",
    "JUPYTER_CONFIG_DIR": str(PROJECT_ROOT / ".jupyter"),
    "PROJECT_ROOT": str(PROJECT_ROOT),
    "DATA_DIR": str(DATA_DIR),
    "MODELS_DIR": str(MODELS_DIR),
    "FOOTBALL_ANALYTICS_ENV": "jupyter",
}

# ================================
# CONFIGURACIÓN DE EXTENSIONES
# ================================

# Lista de extensiones recomendadas para Football Analytics
RECOMMENDED_EXTENSIONS = [
    "@jupyter-widgets/jupyterlab-manager",
    "jupyterlab-plotly",
    "@jupyterlab/git",
    "@lckr/jupyterlab_variableinspector",
    "jupyterlab-spreadsheet",
    "@jupyterlab/toc",
]

# ================================
# STARTUP SCRIPTS Y HOOKS
# ================================


def setup_football_analytics_environment():
    """Configura el entorno específico para Football Analytics"""

    # Agregar paths del proyecto al sys.path
    project_paths = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "app"),
        str(PROJECT_ROOT / "app" / "services"),
        str(PROJECT_ROOT / "app" / "utils"),
        str(PROJECT_ROOT / "app" / "ml_models"),
    ]

    for path in project_paths:
        if path not in sys.path:
            sys.path.insert(0, path)

    # Crear directorios necesarios
    required_dirs = [
        NOTEBOOKS_DIR / "exploration",
        NOTEBOOKS_DIR / "modeling",
        NOTEBOOKS_DIR / "analysis",
        NOTEBOOKS_DIR / "reports",
        NOTEBOOKS_DIR / "experiments",
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATA_DIR / "external",
        DATA_DIR / "interim",
        DATA_DIR / "final",
        PROJECT_ROOT / "exports",
        PROJECT_ROOT / "cache" / "notebooks",
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # Configurar logging para notebooks
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(PROJECT_ROOT / "logs" / "jupyter.log"),
            logging.StreamHandler(),
        ],
    )


# Ejecutar configuración al inicio
setup_football_analytics_environment()

# ================================
# CONFIGURACIÓN DE PLOTTING
# ================================

# Configuración por defecto para matplotlib
c.InlineBackend.rc = {
    "figure.figsize": (12, 8),
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Formato de figuras inline
c.InlineBackend.figure_formats = {"png", "retina"}

# ================================
# CONFIGURACIÓN DE AUTOCOMPLETADO
# ================================

# Configuración avanzada de autocompletado
c.Completer.use_jedi = True
c.Completer.greedy = True

# ================================
# CONFIGURACIÓN DE SEGURIDAD
# ================================

# Configuración de Content Security Policy
c.ServerApp.tornado_settings = {
    "headers": {
        "Content-Security-Policy": "frame-ancestors 'self'; report-uri /api/security/csp-report",
    }
}

# ================================
# STARTUP HOOKS Y SCRIPTS
# ================================


def pre_save_hook(model, **kwargs):
    """Hook ejecutado antes de guardar notebooks"""
    if model["type"] == "notebook":
        # Limpiar outputs si es necesario
        if "metadata" in model.get("content", {}):
            # Agregar metadata del proyecto
            model["content"]["metadata"]["football_analytics"] = {
                "project": PROJECT_NAME,
                "version": "2.1.0",
                "created_with": "Football Analytics Jupyter Config",
            }


# Registrar hooks
c.FileContentsManager.pre_save_hook = pre_save_hook

# ================================
# CONFIGURACIÓN DE TEMPLATES
# ================================

# Template personalizado para nuevos notebooks
NOTEBOOK_TEMPLATE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏈 Football Analytics - Notebook\n",
                "\n",
                "**Proyecto:** Football Analytics  \n",
                "**Fecha:** $(date)  \n",
                "**Autor:** [Tu nombre]  \n",
                "\n",
                "## Descripción\n",
                "[Describe el objetivo de este notebook]\n",
                "\n",
                "## Datos\n",
                "[Describe los datos que vas a usar]\n",
                "\n",
                "## Metodología\n",
                "[Describe el enfoque que vas a seguir]\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Configuración inicial\n",
                "import sys\n",
                "import os\n",
                "from pathlib import Path\n",
                "\n",
                "# Agregar paths del proyecto\n",
                "PROJECT_ROOT = Path.cwd()\n",
                "if str(PROJECT_ROOT) not in sys.path:\n",
                "    sys.path.append(str(PROJECT_ROOT))\n",
                "    sys.path.append(str(PROJECT_ROOT / 'app'))\n",
                "\n",
                "print(f'📁 Directorio de trabajo: {PROJECT_ROOT}')\n",
                "print(f'🐍 Python path configurado')\n",
                "print(f'🚀 Football Analytics environment ready!')",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Imports básicos para Football Analytics\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import plotly.express as px\n",
                "import plotly.graph_objects as go\n",
                "from datetime import datetime, timedelta\n",
                "\n",
                "# ML imports\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.metrics import accuracy_score, classification_report\n",
                "import xgboost as xgb\n",
                "import lightgbm as lgb\n",
                "\n",
                "# Football Analytics específicos (descomentar según necesites)\n",
                "# from app.services.predictor import PredictorService\n",
                "# from app.services.data_collector import DataCollectorService\n",
                "# from app.utils.helpers import *\n",
                "# from app.utils.constants import *\n",
                "\n",
                "# Configuración de visualización\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette('husl')\n",
                "%matplotlib inline\n",
                "%config InlineBackend.figure_format = 'retina'\n",
                "\n",
                "# Configuración pandas\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)\n",
                "pd.set_option('display.float_format', '{:.3f}'.format)\n",
                "\n",
                "print('✅ Imports y configuración completados')",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0",
        },
        "football_analytics": {
            "project": "Football Analytics",
            "version": "2.1.0",
            "template": "default",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

# ================================
# CONFIGURACIÓN DE LOGGING
# ================================

# Configuración de logs específica para Jupyter
c.Application.log_level = "INFO"
c.Application.log_format = (
    "[%(levelname)1.1s %(asctime)s.%(msecs).03d %(name)s] %(message)s"
)
c.Application.log_datefmt = "%Y-%m-%d %H:%M:%S"

# Log file para Jupyter
c.FileNotebookManager.log = True

# ================================
# CONFIGURACIÓN DE PERFORMANCE
# ================================

# Configuración de memoria y performance
c.ResourceUseDisplay.mem_limit = 2 * 1024**3  # 2GB
c.ResourceUseDisplay.track_cpu_percent = True
c.ResourceUseDisplay.cpu_limit = 1.0

# ================================
# CONFIGURACIÓN DE EXPORTACIÓN
# ================================

# Configuración para nbconvert
c.Exporter.template_paths = [str(PROJECT_ROOT / "templates")]

# Configuración de exportación HTML
c.HTMLExporter.exclude_input_prompt = True
c.HTMLExporter.exclude_output_prompt = True

# ================================
# CONFIGURACIÓN DE GIT INTEGRATION
# ================================

# Configuración para integración con Git
c.GitHubConfig.access_token = os.environ.get("GITHUB_TOKEN", "")

# ================================
# FUNCIONES UTILITARIAS
# ================================


def create_analysis_notebook(name, description=""):
    """Crea un nuevo notebook para análisis"""
    import json
    from datetime import datetime

    # Personalizar template
    notebook = NOTEBOOK_TEMPLATE.copy()
    notebook["cells"][0]["source"] = (
        notebook["cells"][0]["source"]
        .replace("[Describe el objetivo de este notebook]", description)
        .replace("$(date)", datetime.now().strftime("%Y-%m-%d"))
    )

    # Guardar notebook
    notebook_path = NOTEBOOKS_DIR / "analysis" / f"{name}.ipynb"
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"📓 Notebook creado: {notebook_path}")
    return notebook_path


def load_football_data():
    """Función de conveniencia para cargar datos de fútbol"""
    try:
        from app.utils.helpers import load_sample_data

        return load_sample_data()
    except ImportError:
        print("⚠️ No se pudo importar helpers. Usando datos de ejemplo...")
        return pd.DataFrame(
            {
                "team": ["Real Madrid", "Barcelona", "Manchester City"],
                "goals": [2.1, 2.3, 2.0],
                "league": ["La Liga", "La Liga", "Premier League"],
            }
        )


# ================================
# STARTUP MESSAGE
# ================================


def display_startup_message():
    """Muestra mensaje de bienvenida"""
    message = f"""
🏈 FOOTBALL ANALYTICS - JUPYTER ENVIRONMENT
{'='*60}
📦 Proyecto: {PROJECT_NAME} v2.1.0
📁 Directorio: {PROJECT_ROOT}
📊 Notebooks: {NOTEBOOKS_DIR}
💾 Datos: {DATA_DIR}
🤖 Modelos: {MODELS_DIR}

🚀 Funciones disponibles:
   - create_analysis_notebook(name, description)
   - load_football_data()
   - Todos los servicios de Football Analytics

🔗 URLs útiles:
   - Jupyter Lab: http://localhost:8888/lab
   - API Docs: http://localhost:8000/docs (si está ejecutándose)

✨ ¡Happy analyzing! ⚽
{'='*60}
    """
    print(message)


# Mostrar mensaje al inicio
display_startup_message()

# ================================
# CONFIGURACIÓN FINAL
# ================================

# Configuración de shutdown
c.ServerApp.shutdown_no_activity_timeout = 7200  # 2 horas de inactividad

# Configuración de backup automático
c.FileCheckpoints.checkpoint_dir = str(
    PROJECT_ROOT / ".jupyter" / "checkpoints"
)

# ================================
# NOTAS Y DOCUMENTACIÓN
# ================================

"""
NOTAS DE USO:

1. INICIAR JUPYTER CON ESTA CONFIGURACIÓN:
   jupyter lab --config=jupyter_config.py

2. VARIABLES DE ENTORNO DISPONIBLES:
   - PROJECT_ROOT: Raíz del proyecto
   - DATA_DIR: Directorio de datos
   - MODELS_DIR: Directorio de modelos
   - FOOTBALL_ANALYTICS_ENV: Identificador del entorno

3. FUNCIONES UTILITARIAS:
   - create_analysis_notebook(): Crea notebooks con template
   - load_football_data(): Carga datos de ejemplo
   - setup_football_analytics_environment(): Configura entorno

4. EXTENSIONES RECOMENDADAS:
   - Variable Inspector: Para debugging
   - Git Integration: Para control de versiones
   - Plotly: Para visualizaciones interactivas
   - Table of Contents: Para navegación

5. ESTRUCTURA DE NOTEBOOKS:
   - notebooks/exploration/: Análisis exploratorio
   - notebooks/modeling/: Desarrollo de modelos
   - notebooks/analysis/: Análisis específicos
   - notebooks/reports/: Reportes finales

6. INTEGRACIÓN CON EL PROYECTO:
   - Acceso directo a servicios de Football Analytics
   - Variables de entorno configuradas
   - Paths del proyecto en sys.path
   - Logging integrado

7. COMANDOS ÚTILES EN NOTEBOOKS:
   %load_ext autoreload
   %autoreload 2
   %time your_function()
   %memit your_function()
   !pip install package_name

8. EXPORTACIÓN:
   - HTML: jupyter nbconvert --to html notebook.ipynb
   - PDF: jupyter nbconvert --to pdf notebook.ipynb
   - Python: jupyter nbconvert --to python notebook.ipynb

9. SEGURIDAD:
   - No usar en producción sin token/password
   - Configurar CORS apropiadamente
   - Revisar permisos de archivos

10. PERFORMANCE:
    - Límite de memoria: 2GB
    - Timeout de kernels: 60s
    - Archivos máximos: 50MB
"""
