# README - Guía de Desarrollo para Agentes de IA

## Flujo de trabajo

Cada spec en este directorio es una unidad de trabajo independiente para ser ejecutada por un agente de IA.

### Reglas de oro

1. **Una spec a la vez.** No mezcles implementación de múltiples specs en la misma sesión.
2. **Sigue el orden de dependencias.** Revisa la sección `## Dependencies` antes de empezar.
3. **Tests primero, luego código.** Los criterios de aceptación son los tests.
4. **Git activo.** Haz commits atómicos con el prefijo de la spec.
5. **SIEMPRE branch → commit → push → verificar → merge a main.**

### Workflow git OBLIGATORIO por spec

```bash
# ═══════════════════════════════════════════════════════════════════
# PASO 1: Crear feature branch desde main actualizado
# ═══════════════════════════════════════════════════════════════════
git checkout main
git pull origin main
git checkout -b feature/spec-XXX-nombre-corto

# ═══════════════════════════════════════════════════════════════════
# PASO 2: Implementar con commits atómicos y push frecuente
# ═══════════════════════════════════════════════════════════════════
git add <archivos>
git commit -m "feat(SPEC-XXX): descripcion del cambio"
git push origin feature/spec-XXX-nombre-corto   # <-- OBLIGATORIO

# ═══════════════════════════════════════════════════════════════════
# PASO 3: Verificar que TODO funciona
# ═══════════════════════════════════════════════════════════════════
pytest tests/ -v --tb=short
ruff check src/ tests/
mypy src/

# ═══════════════════════════════════════════════════════════════════
# PASO 4: Merge a main SOLO cuando CI está verde y tests pasan
# ═══════════════════════════════════════════════════════════════════
git checkout main
git pull origin main                     # Actualizar por si hubo cambios
git merge feature/spec-XXX-nombre-corto  # Fast-forward si es posible
git push origin main                     # <-- OBLIGATORIO

# ═══════════════════════════════════════════════════════════════════
# PASO 5: Limpiar branch (opcional)
# ═══════════════════════════════════════════════════════════════════
git branch -d feature/spec-XXX-nombre-corto
git push origin --delete feature/spec-XXX-nombre-corto
```

### NO hacer

- ❌ Commits directos a `main`
- ❌ Merge sin haber hecho push del branch antes
- ❌ Merge sin tests verdes
- ❌ Amend de commits ya pusheados (reescribe historia remota)
- ❌ `git push --force` a `main`

### Convención de commits

```
feat(SPEC-XXX): description for new feature
fix(SPEC-XXX): description for bugfix
test(SPEC-XXX): description for test addition
refactor(SPEC-XXX): description for refactor
docs(SPEC-XXX): description for documentation
chore(SPEC-XXX): description for tooling, deps, config
```

### Estructura de una spec

Cada archivo `SPEC-XXX-nombre.md` contiene:

- **Status**: PLANNED | IN_PROGRESS | COMPLETED
- **Objective**: Qué debe lograr el código
- **Dependencies**: Qué specs deben estar completas antes
- **Technical Design**: Clases, funciones, tipos esperados
- **Acceptance Criteria**: Tests que definen "done"
- **Files to Create/Modify**: Lista exacta de archivos
- **Git Workflow**: Branches y commits esperados con push explícito

### Orden de ejecución recomendado

```
001-project-setup        [COMPLETED]
    └── 002-database-layer    [COMPLETED]
         ├── 003-dixon-coles-model   [COMPLETED]
         │    └── 004-expected-score-calculator  [COMPLETED]
         │         └── 005-strategy-selector     [COMPLETED]
         ├── 006-odds-api-etl       [COMPLETED]
         ├── 007-api-football-etl   [COMPLETED]
          └── 008-data-pipeline      [COMPLETED] (deps: 006, 007)
               └── 009-gradient-boost-model   [COMPLETED]
                   └── 010-model-ensemble    [COMPLETED] (deps: 003, 009)
    ├── 011-polla-scraper   [COMPLETED]
│         └── 012-game-theory   [COMPLETED] (deps: 011)
    │         └── 013-bracket-optimizer  [COMPLETED] (deps: 004, 012)
    └── 014-monte-carlo-simulator   [COMPLETED] (deps: 003, 010)
         └── 015-backtesting   [COMPLETED] (deps: 003, 004, 014)
    └── 016-cli-interface   [COMPLETED] (deps: 003, 004, 005)
         └── 017-fastapi-backend   [PLANNED] (deps: 002, 003, 004, 005, 014, 015)
              └── 018-web-ui (Streamlit)  [PLANNED] (deps: 017)
```
