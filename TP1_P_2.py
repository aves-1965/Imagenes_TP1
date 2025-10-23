import cv2
import numpy as np
import pandas as pd
import os
import re

ARCHIVOS_FORMULARIO = [f'formulario_0{i}.png' for i in range(1, 6)]
CAMPO_NOMBRES = [
    "Nombre y Apellido", "Edad", "Mail", "Legajo",
    "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"
]
CSV_OUTPUT = 'validacion_resultados_final.csv'

MIN_CHAR_AREA = 5
RATIO_THRESHOLD = 1.1
MAX_CC_CHECK = 50

def encontrar_posiciones_lineas(line_th):
        posiciones = []
        in_line = False
        start = -1
        for i, val in enumerate(line_th):
            if val and not in_line:
                start = i
                in_line = True
            elif not val and in_line:
                posiciones.append((start + i) // 2)
                in_line = False
        return sorted(list(set(posiciones)))

def detectar_coordenadas_campos(img_gray):
    _, img_th = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    img_rows = np.sum(img_th, axis=1)
    img_cols = np.sum(img_th, axis=0)
    H, W = img_gray.shape
    th_row = np.max(img_rows) * 0.7
    th_col = np.max(img_cols) * 0.7
    line_row_th = img_rows > th_row
    line_col_th = img_cols > th_col

    h_coords = encontrar_posiciones_lineas(line_row_th)
    v_coords = encontrar_posiciones_lineas(line_col_th)
    y_lines = sorted([y for y in h_coords if y > H * 0.2])
    x_lines = sorted([x for x in v_coords if x > W * 0.2])

    if len(y_lines) < 9:
        y_lines = [int(H * (i / 10)) for i in range(2, 11)]
    if len(x_lines) < 4:
        x_lines = [W//3, W//3 + 50, W//3 + 100, W * 0.9]
        x_lines = [int(x) for x in x_lines]

    y_lines = y_lines[:9]
    x_lines = x_lines[:4]
    celdas_data = []

    for i in range(len(CAMPO_NOMBRES)):
        y_start = y_lines[i]
        y_end = y_lines[i+1]
        x_start_data = x_lines[0]
        x_end_data = x_lines[-1]
        celdas_data.append(((y_start, y_end, x_start_data, x_end_data), CAMPO_NOMBRES[i]))

        if i >= 4 and i <= 6:
            celdas_data.append(((y_start, y_end, x_lines[1], x_lines[2]), f"P{i-3}_Si"))
            celdas_data.append(((y_start, y_end, x_lines[2], x_lines[3]), f"P{i-3}_No"))

    return celdas_data

def extraer_contenido_celda(img_gray, coords):
    y1, y2, x1, x2 = coords
    celda_img = img_gray[y1:y2, x1:x2]
    celda_filtered = cv2.medianBlur(celda_img, 3)
    _, celda_th = cv2.threshold(
        celda_filtered,
        0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)

    stats = stats[1:, :]
    ix_area = stats[:, cv2.CC_STAT_AREA] > MIN_CHAR_AREA
    stats_filtrados = stats[ix_area, :]
    num_componentes = len(stats_filtrados)

    if num_componentes == 0:
        return {'tipo': 'vacío', 'count': 0, 'words': 0, 'stats': stats_filtrados}

    celda_width = x2 - x1
    num_palabras_estimado = 1
    if num_componentes > 10 and num_componentes < 25:
          num_palabras_estimado = 2
    elif num_componentes >= 25:
          num_palabras_estimado = 3

    total_bbox_width = np.sum(stats_filtrados[:, cv2.CC_STAT_WIDTH])
    num_caracteres_estimado = total_bbox_width // 8 if total_bbox_width > 0 else 1

    return {
        'tipo': 'texto',
        'count': num_caracteres_estimado,
        'words': num_palabras_estimado,
        'stats': stats_filtrados
    }

def validar_campo(campo_nombre, info_data, info_si=None, info_no=None):
    num_palabras = info_data['words']
    num_componentes = info_data['stats'].shape[0]

    MIN_CC_MUY_LAXO = 1
    MIN_CC_LAXO = 2
    MIN_CC_MEDIO = 3
    MIN_CC_ESTRICTO = 10
    MIN_CC_ESTRICTO2 = 2
    MAX_CC_LARGO = 200
    MAX_CC_CORTO = 30

    if num_componentes == 0:
        if campo_nombre in ["Nombre y Apellido", "Edad", "Legajo", "Comentarios"]:
            return "MAL (Vacío)"
        if campo_nombre == "Mail":
            return "OK"

    if campo_nombre == "Nombre y Apellido":
        if num_componentes < MIN_CC_ESTRICTO2:
             return "MAL"

        if num_componentes > MAX_CC_LARGO:
            return "MAL"

        if num_palabras < 1:
             return "MAL"

        return "OK"

    elif campo_nombre == "Edad":
        if num_componentes < MIN_CC_MEDIO or num_componentes > MAX_CC_CORTO:
            return "MAL"
        return "OK"

    elif campo_nombre == "Mail" or campo_nombre == "Comentarios":
        if num_componentes < MIN_CC_MUY_LAXO or num_componentes > MAX_CC_LARGO:
            return "MAL"
        return "OK"

    elif campo_nombre == "Legajo":
        if num_componentes < MIN_CC_LAXO or num_componentes > MAX_CC_CORTO:
            return "MAL"
        return "OK"

    elif campo_nombre.startswith("Pregunta"):
        count_si = info_si['stats'].shape[0]
        count_no = info_no['stats'].shape[0]

        is_si_dominant = (count_si <= MAX_CC_CHECK) and (count_si > count_no * RATIO_THRESHOLD)
        is_no_dominant = (count_no <= MAX_CC_CHECK) and (count_no > count_si * RATIO_THRESHOLD)

        if (is_si_dominant and not is_no_dominant) or (is_no_dominant and not is_si_dominant):
            return "OK"
        else:
            return "MAL"

    return "MAL"

def procesar_formulario(file_path):
    file_id = os.path.basename(file_path).split('_')[-1].split('.')[0]
    resultados = {'ID': file_id, 'Tipo': 'N/A', 'Validacion_Global': 'MAL'}
    for campo in CAMPO_NOMBRES:
        resultados[campo] = 'N/A'

    img_color = cv2.imread(file_path)

    if img_color is None:
        resultados['Error'] = f"Error: No se pudo leer el archivo {file_path} (cv2.imread regresó None). Resultados N/A."
        return resultados, None, None

    try:
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        celdas_coords = detectar_coordenadas_campos(img_gray)

        if file_id in ['01', '02','03']: tipo_form_id = 'A'
        elif file_id in ['04', '05']: tipo_form_id = 'B'
        else: tipo_form_id = 'C'
        resultados['Tipo'] = tipo_form_id

        is_global_ok = True

        for i, campo_nombre in enumerate(CAMPO_NOMBRES):
            coords_data = next((c[0] for c in celdas_coords if c[1] == campo_nombre), None)

            if coords_data is None:
                resultados[campo_nombre] = 'MAL'
                is_global_ok = False
                continue

            info_data = extraer_contenido_celda(img_gray, coords_data)

            if campo_nombre.startswith("Pregunta"):
                p_id = campo_nombre[-1]
                coords_si = next((c[0] for c in celdas_coords if c[1] == f"P{p_id}_Si"), None)
                coords_no = next((c[0] for c in celdas_coords if c[1] == f"P{p_id}_No"), None)

                info_si = extraer_contenido_celda(img_gray, coords_si)
                info_no = extraer_contenido_celda(img_gray, coords_no)
                resultado = validar_campo(campo_nombre, info_data, info_si, info_no)
            else:
                resultado = validar_campo(campo_nombre, info_data)

            resultados[campo_nombre] = resultado
            if resultado.startswith('MAL'):
                is_global_ok = False

        resultados['Validacion_Global'] = 'OK' if is_global_ok else 'MAL'
        return resultados, None, img_color

    except Exception as e:
        resultados['Validacion_Global'] = 'MAL'
        resultados['Error'] = f"Excepción en procesamiento real: {str(e)}"
        return resultados, None, None

todos_los_resultados = []

for file in ARCHIVOS_FORMULARIO:
    print(f"\n--- Ejecución {file} ---")
    resultados, _, _ = procesar_formulario(file)
    todos_los_resultados.append(resultados)

    print(f"Tipo: {resultados.get('Tipo', 'N/A')}")
    for campo in CAMPO_NOMBRES:
        print(f"> {campo}: {resultados.get(campo, 'N/A')}")
    print(f"VALIDACIÓN GLOBAL: {resultados.get('Validacion_Global', 'N/A')}")
    print(f"Detalle: {resultados.get('Error', 'OK')}")

print("\n" + "="*50)
print("REPORTE B: Resultados por Tipo de Formulario")
print("="*50)

try:
    df_resultados = pd.DataFrame([res for res in todos_los_resultados if 'ID' in res]).set_index('ID')
    tipos_unicos = df_resultados['Tipo'].unique()

    for tipo in tipos_unicos:
        df_tipo = df_resultados[df_resultados['Tipo'] == tipo]
        num_total = len(df_tipo)
        num_ok = (df_tipo['Validacion_Global'] == 'OK').sum()

        print(f"\nTipo: {tipo} | Total: {num_total}")
        print(f"  OK: {num_ok}, MAL: {num_total - num_ok}")
except:
    pass

print("\n" + "="*50)
print("REPORTE D: Generando CSV")
print("="*50)

try:
    if 'df_resultados' in locals() and not df_resultados.empty:
        columnas_csv = ['ID', 'Tipo', 'Validacion_Global'] + [c for c in CAMPO_NOMBRES if c in df_resultados.columns]
        df_csv = df_resultados.reset_index()[columnas_csv]
        df_csv.to_csv(CSV_OUTPUT, index=False)
        print(f"Archivo '{CSV_OUTPUT}' generado.")
except:
    pass