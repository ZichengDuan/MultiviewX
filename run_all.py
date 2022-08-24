from calibrateCamera import calibrate
from generatePOM import generate_POM
from generateAnnotation import annotate
from generateAnnotationOutside import annotate_outside
if __name__ == '__main__':
    calibrate()
    generate_POM()
    annotate()
    annotate_outside()
