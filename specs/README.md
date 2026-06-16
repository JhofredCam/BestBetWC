# README - Guía de Desarrollo para Agentes de IA

## Flujo de trabajo

Cada spec en este directorio es una unidad de trabajo independiente para ser ejecutada por un agente de IA.

### Reglas de oro

1. **Una spec a la vez.** No mezcles implementación de múltiples specs en la misma sesión.
2. **Sigue el orden de dependencias.** Revisa la sección `## Dependencies` antes de empezar.
3. **Tests primero, luego código.** Los criterios de aceptación son los tests.
4. **Git activo.** Haz commits atómicos con el prefijo de la spec.

### Workflow git por spec

```bash
# 1. Crear branch desde main
git checkout main
git pull origin main
git checkout -b feature/spec-XXX-nombre-corto

# 2. Implementar (haciendo commits frecuentes)
git add <archivos>
git commit -m "feat(SPEC-XXX): descripcion del cambio"

# 3. Verificar
pytest tests/ -v
ruff check src/ tests/

# 4. Merge a main (solo si el usuario lo aprueba)
git checkout main
git merge feature/spec-XXX-nombre-corto
```

### Convención de commits

```
feat(SPEC-XXX): description for new feature
fix(SPEC-XXX): description for bugfix
test(SPEC-XXX): description for test addition
refactor(SPEC-XXX): description for refactor
docs(SPEC-XXX): description for documentation
```

### Estructura de una spec

Cada archivo `SPEC-XXX-nombre.md` contiene:

- **Status**: PLANNED | IN_PROGRESS | COMPLETED
- **Objective**: Qué debe lograr el código
- **Dependencies**: Qué specs deben estar completas antes
- **Technical Design**: Clases, funciones, tipos esperados
- **Acceptance Criteria**: Tests que definen "done"
- **Files to Create/Modify**: Lista exacta de archivos
- **Git Workflow**: Branches y commits esperados

### Orden de ejecución recomendado

```
001-project-setup        [COMPLETED - documentar]
    └── 002-database-layer    [PLANNED]
         ├── 003-dixon-coles-model   [COMPLETED]
         │    └── 004-expected-score-calculator  [COMPLETED]
         │         └── 005-strategy-selector     [COMPLETED]
         ├── 006-odds-api-etl       [PLANNED]
         ├── 007-api-football-etl   [PLANNED]
         └── 008-data-pipeline      [PLANNED] (deps: 006, 007)
              └── 009-gradient-boost-model   [PLANNED]
                   └── 010-model-ensemble    [PLANNED] (deps: 003, 009)
    ├── 011-polla-scraper   [PLANNED]
    │    └── 012-game-theory   [PLANNED] (deps: 011)
    │         └── 013-bracket-optimizer  [PLANNED] (deps: 004, 012)
    └── 014-monte-carlo-simulator   [PLANNED] (deps: 003, 010)
         └── 015-backtesting   [PLANNED] (deps: 003, 004, 014)
    └── 016-cli-interface   [COMPLETED] (deps: 003, 004, 005)
```
