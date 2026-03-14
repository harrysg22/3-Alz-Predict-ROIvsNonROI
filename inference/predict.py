"""
predict.py  -  Script de inferencia para diagnostico de Alzheimer

DESCRIPCION:
    Toma los archivos de salida de FastSurfer para UN paciente nuevo,
    aplica el mismo preprocesado que el notebook de entrenamiento,
    carga el modelo entrenado (.h5) y devuelve el diagnostico.

REQUISITOS:
    - Haber entrenado el modelo y tener el archivo .h5 guardado.
    - El paciente nuevo debe haber sido procesado previamente con FastSurfer
      (igual que los datos de entrenamiento).
    - Paquetes: tensorflow, nibabel, scikit-image, opencv-python, numpy

USO EN WINDOWS (Anaconda Prompt):
    conda activate alzheimer
    python inference\\predict.py ^
        --aseg  C:\\ruta\\paciente\\mri\\aparc.DKTatlas+aseg.deep.mgz ^
        --orig  C:\\ruta\\paciente\\mri\\orig.mgz ^
        --model UpgradedFilter_ROI_ADNI_ADvsMCI.h5

USO EN MAC/LINUX:
    python inference/predict.py \\
        --aseg  /ruta/paciente/mri/aparc.DKTatlas+aseg.deep.mgz \\
        --orig  /ruta/paciente/mri/orig.mgz \\
        --model UpgradedFilter_ROI_ADNI_ADvsMCI.h5

    Tambien puedes pasar la carpeta raiz del sujeto directamente:
    python inference/predict.py \\
        --subject_dir /ruta/paciente \\
        --model UpgradedFilter_ROI_ADNI_ADvsMCI.h5

NOTA PARA MAC M4:
    En lugar de tensorflow normal, instala:
        pip install tensorflow-macos tensorflow-metal
    El resto del script funciona igual.

SALIDA ESPERADA:
    ========================================
    RESULTADO DEL DIAGNOSTICO
    ========================================
    AD  (Alzheimer)         : 87.3%
    MCI (Deterioro leve)    : 12.7%

    Diagnostico mas probable: AD (Alzheimer)
    ========================================
    ADVERTENCIA: Este resultado es solo orientativo.
    No reemplaza el criterio de un medico especialista.
    ========================================
"""

import argparse
import os
import sys

import numpy as np


# ============================================================
# Funciones de preprocesado (identicas al notebook)
# ============================================================

def apply_mask(aseg_image, brain_image, labels=None):
    """
    Enmascara el MRI original para quedarse solo con las 6 ROIs:
      - 17, 53: hipocampo izq/der
      - 2,  41: sustancia blanca cerebral izq/der
      - 7,  46: sustancia blanca cerebelar izq/der
    """
    if labels is None:
        labels = [17, 53, 2, 7, 41, 46]

    aseg_data = aseg_image.get_fdata()
    origin_data = brain_image.get_fdata()

    brain_mask = np.zeros_like(aseg_data)
    for label in labels:
        brain_mask += np.where(aseg_data == label, 1, 0)

    return origin_data * brain_mask


def enhance_slice(slice_data):
    """CLAHE (mejora de contraste) en una sola capa 2D."""
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(slice_data.astype(np.uint8))


def enhance_image(img_data):
    """Aplica CLAHE capa a capa en el eje Z."""
    enhanced = np.zeros_like(img_data)
    for i in range(img_data.shape[2]):
        enhanced[:, :, i] = enhance_slice(img_data[:, :, i])
    return enhanced


def sharpen_image(image, strength=1.0):
    """Enfoque (unsharp mask)."""
    from skimage.filters import unsharp_mask
    return unsharp_mask(image, radius=1, amount=strength)


def preprocess(aseg_path, orig_path, target_shape=(100, 100, 100)):
    """
    Pipeline de preprocesado completo para un sujeto:
      1. Cargar aseg y orig de FastSurfer
      2. Aplicar mascara de ROIs
      3. Redimensionar al tamano objetivo
      4. CLAHE
      5. Sharpening
      6. Agregar dimension de canal
    """
    import nibabel
    from skimage.transform import resize

    aseg_image = nibabel.load(aseg_path)
    orig_image = nibabel.load(orig_path)

    image = apply_mask(aseg_image, orig_image)
    image = resize(image, target_shape, anti_aliasing=True)
    image = enhance_image(image)
    image = sharpen_image(image)

    # Agregar batch y canal: (1, 100, 100, 100, 1)
    image = np.expand_dims(image, axis=0)   # batch
    image = np.expand_dims(image, axis=-1)  # canal

    return image


# ============================================================
# Funcion principal de prediccion
# ============================================================

CLASS_NAMES = {
    0: "AD  (Alzheimer)",
    1: "MCI (Deterioro leve)",
}


def predict(aseg_path, orig_path, model_path, target_shape=(100, 100, 100)):
    """
    Carga el modelo y predice para un sujeto.
    Devuelve un dict con probabilidades por clase.
    """
    print("\nCargando modelo...")
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"\nERROR al cargar el modelo: {e}")
        print("\nVerifica que:")
        print("  - El archivo .h5 existe en la ruta indicada")
        print("  - Tienes instalado tensorflow (o tensorflow-macos en Mac M4)")
        sys.exit(1)

    print("Preprocesando imagen...")
    try:
        image = preprocess(aseg_path, orig_path, target_shape)
    except Exception as e:
        print(f"\nERROR al preprocesar la imagen: {e}")
        print("\nVerifica que:")
        print("  - Los archivos .mgz existen y no estan corruptos")
        print("  - Tienes instalados: nibabel, opencv-python, scikit-image")
        sys.exit(1)

    print("Realizando prediccion...")
    probabilities = model.predict(image, verbose=0)[0]

    results = {i: float(p) for i, p in enumerate(probabilities)}
    return results


# ============================================================
# Presentacion del resultado
# ============================================================

def print_results(results):
    best_class = max(results, key=results.get)
    best_name = CLASS_NAMES.get(best_class, f"Clase {best_class}")

    print()
    print("=" * 50)
    print("  RESULTADO DEL DIAGNOSTICO")
    print("=" * 50)
    for cls_idx, prob in sorted(results.items(), key=lambda x: -x[1]):
        name = CLASS_NAMES.get(cls_idx, f"Clase {cls_idx}")
        bar = "#" * int(prob * 30)
        print(f"  {name:25s}: {prob*100:5.1f}%  {bar}")
    print()
    print(f"  Diagnostico mas probable: {best_name}")
    print("=" * 50)
    print("  ADVERTENCIA: Este resultado es solo orientativo.")
    print("  No reemplaza el criterio de un medico especialista.")
    print("=" * 50)
    print()


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostico de Alzheimer para un paciente nuevo"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--subject_dir",
        help=(
            "Carpeta raiz del sujeto procesado por FastSurfer. "
            "Se asume que contiene mri/orig.mgz y "
            "mri/aparc.DKTatlas+aseg.deep.mgz"
        )
    )
    input_group.add_argument(
        "--aseg",
        help="Ruta al archivo aparc.DKTatlas+aseg.deep.mgz"
    )

    parser.add_argument(
        "--orig",
        help="Ruta al archivo orig.mgz (requerido si se usa --aseg)"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ruta al archivo .h5 del modelo entrenado"
    )
    parser.add_argument(
        "--target_shape",
        default="100,100,100",
        help="Forma objetivo del volumen, separada por comas (default: 100,100,100)"
    )

    args = parser.parse_args()

    # Resolver rutas de entrada
    if args.subject_dir:
        mri_dir = os.path.join(args.subject_dir, "mri")
        aseg_path = os.path.join(mri_dir, "aparc.DKTatlas+aseg.deep.mgz")
        orig_path = os.path.join(mri_dir, "orig.mgz")
    else:
        aseg_path = args.aseg
        if not args.orig:
            parser.error("--orig es requerido cuando se usa --aseg")
        orig_path = args.orig

    # Validar que existan los archivos
    for path, name in [(aseg_path, "aseg"), (orig_path, "orig"), (args.model, "model")]:
        if not os.path.exists(path):
            print(f"\nERROR: No se encontro el archivo {name}:")
            print(f"  {path}")
            sys.exit(1)

    # Parsear target_shape
    try:
        target_shape = tuple(int(x) for x in args.target_shape.split(","))
        assert len(target_shape) == 3
    except Exception:
        print("ERROR: --target_shape debe tener 3 valores separados por coma (ej: 100,100,100)")
        sys.exit(1)

    print("\nArchivos de entrada:")
    print(f"  ASEG:   {aseg_path}")
    print(f"  ORIG:   {orig_path}")
    print(f"  Modelo: {args.model}")
    print(f"  Shape:  {target_shape}")

    results = predict(aseg_path, orig_path, args.model, target_shape)
    print_results(results)


if __name__ == "__main__":
    main()
