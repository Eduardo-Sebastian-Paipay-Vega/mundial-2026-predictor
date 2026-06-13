# Deploy en Render.com — Mundial 2026 Predictor API

URL esperada: **https://mundial-predictor-api.onrender.com**

---

## Prerrequisitos

- Cuenta en [GitHub](https://github.com) y [Render.com](https://render.com)
- Git instalado localmente
- Los archivos `models/` y `data/` deben estar en el repositorio (free tier no tiene almacenamiento externo persistente)

---

## 1. Subir código a GitHub

### 1a. Crear repositorio en GitHub

Ve a https://github.com/new y crea un repo llamado `mundial-predictor` (privado o público, sin inicializar).

### 1b. Inicializar y subir desde la raíz del proyecto (`c:\botas\predic`)

```bash
# Crear .gitignore para excluir entorno virtual y archivos sensibles
cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
.DS_Store
*.log
EOF

# Inicializar repositorio
git init

# Agregar todos los archivos (incluidos models/ y data/ para free tier)
git add .

# Verificar que .env NO está incluido
git status

# Primer commit
git commit -m "feat: deploy inicial — Mundial 2026 Predictor API"

# Conectar con GitHub (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/mundial-predictor.git

# Subir rama principal
git branch -M main
git push -u origin main
```

> **Importante:** Confirma que `.env` no aparece en `git status` antes del commit. El archivo `.env.example` si puede subirse; el `.env` real, nunca.

---

## 2. Crear Web Service en Render.com

1. Entra a [https://dashboard.render.com](https://dashboard.render.com)
2. Click en **"New +"** -> **"Web Service"**
3. En **"Connect a repository"**, selecciona tu repo `mundial-predictor`
   - Si no aparece, haz click en "Configure account" para autorizar el acceso a GitHub
4. Render detectará `render.yaml` automáticamente. Confirma los valores:
   - **Name:** `mundial-predictor-api`
   - **Region:** Oregon (US West) o la más cercana a tu audiencia
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT backend.main:app`
   - **Plan:** `Free`
5. Click en **"Create Web Service"**

---

## 3. Configurar variables de entorno en Render

En el dashboard del servicio, ve a **"Environment"** en el menú lateral izquierdo.

Agrega las siguientes variables (click en **"Add Environment Variable"** por cada una):

| Variable | Valor | Notas |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | `<tu_key>` | De football-data.org |
| `OPENWEATHER_API_KEY` | `<tu_key>` | Opcional; dejar vacío si no se usa |
| `LOG_LEVEL` | `INFO` | Ya definido en render.yaml |
| `ENVIRONMENT` | `production` | Ya definido en render.yaml |
| `PYTHONUNBUFFERED` | `1` | Ya definido en render.yaml |

> Las variables marcadas con `sync: false` en `render.yaml` (`FOOTBALL_DATA_API_KEY` y `OPENWEATHER_API_KEY`) deben ingresarse manualmente aqui — Render no las lee del archivo para proteger secretos.

Tras agregar las variables, click en **"Save Changes"**. Render reiniciará el servicio automáticamente.

---

## 4. Verificar el deployment

### 4a. Ver logs en tiempo real

En el dashboard, tab **"Logs"**. Debes ver:

```
Iniciando FastAPI — cargando modelos...
Backend listo
```

El primer build puede tardar 3-5 minutos mientras instala dependencias.

### 4b. Verificar endpoint /health con curl

```bash
curl https://mundial-predictor-api.onrender.com/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "models_loaded": true,
  "n_features": <numero>,
  "historical_matches": <numero>,
  "version": "1.0.0"
}
```

### 4c. Probar endpoint de prediccion

```bash
curl -X POST https://mundial-predictor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"team_home": "Argentina", "team_away": "Francia", "n_sim": 1000}'
```

### 4d. Ver lista de equipos disponibles

```bash
curl https://mundial-predictor-api.onrender.com/teams
```

---

## 5. Consideraciones del free tier

- **Cold start:** El servicio se "duerme" tras 15 minutos de inactividad. La primera request luego de inactividad tarda ~30 segundos en responder (Render despierta el contenedor y FastAPI carga los modelos).
- **Recursos:** 512 MB RAM, 0.1 CPU. Los modelos `.pkl` deben caber en ese limite.
- **Disco efimero:** Los archivos escritos en tiempo de ejecucion (logs, cache) se pierden al reiniciar. Los datos de `data/` y `models/` incluidos en el repo persisten porque estan en el filesystem del contenedor desplegado.
- **CORS:** Si conectas un frontend (ej. Streamlit en otro servicio), actualiza `allow_origins` en `backend/main.py` con la URL de produccion del frontend.

---

## 6. Actualizar el deployment

Cada push a `main` dispara un redeploy automatico en Render:

```bash
git add .
git commit -m "fix: descripcion del cambio"
git push origin main
```

Render detecta el push y ejecuta build + deploy sin intervencion manual.
