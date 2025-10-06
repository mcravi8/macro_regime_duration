"""
Project structure builder for Macro Regime Duration framework.
"""

import os
from pathlib import Path

def create_structure():
    """Create project directory structure"""
    
    directories = [
        'data/raw',
        'data/processed',
        'src/data_collection',
        'src/regime_model',
        'src/yield_curve',
        'src/portfolio_opt',
        'src/utils',
        'notebooks',
        'output/figures',
        'output/tables',
        'output/results',
        'paper/figures',
        'presentation',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        if directory.startswith('src/'):
            (Path(directory) / '__init__.py').touch()
        
        if any(x in directory for x in ['data/', 'output/']):
            (Path(directory) / '.gitkeep').touch()
    
    Path('src/__init__.py').touch()

def create_env_template():
    """Create environment template"""
    content = "FRED_API_KEY=your_fred_api_key_here\n"
    Path('.env.template').write_text(content)

def create_gitignore():
    """Create .gitignore"""
    content = """__pycache__/
*.py[cod]
.Python
venv/
env/
*.egg-info/
data/raw/*.csv
data/processed/*.csv
output/figures/*.png
output/tables/*.csv
output/results/*.txt
.env
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
!**/.gitkeep
"""
    Path('.gitignore').write_text(content)

def main():
    create_structure()
    create_env_template()
    create_gitignore()
    print("Project structure created successfully.")

if __name__ == "__main__":
    main()
