import cv2 as cv
import numpy as np

def get_face(img, img_code=''):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print(f'[ERROR] No faces detected on img {img_code}')
        return None
    
    if len(faces) > 1:
        print(f'[WARNING] {len(faces)} faces detected on img {img_code}')

    # sometimes face detector produces false positives, pick the biggest "face"
    biggest = faces[0]
    for face in faces[1:]:
        if face[2] * face[3] > biggest[2] * biggest[3]:
            biggest = face

    # return biggest face
    (x, y, w, h) = biggest      
    return img[y : y + h, x : x + w]

def generate_faces(show_images=False):
    for img_letter in ['A', 'B', 'C']:
        for img_num in range(10):
            img = cv.imread(f'assets/images/image{img_letter}{img_num}.jpg')
            face = get_face(img, f'{img_letter}{img_num}')
            cv.imwrite(f'assets/faces/face{img_letter}{img_num}.jpg', face)
            
            if show_images:
                cv.imshow(f'img {img_letter}{img_num}', cv.resize(img, None, fx=.5, fy=.5))
                cv.imshow(f'face {img_letter}{img_num}', cv.resize(face, None, fx=.5, fy=.5))
    if show_images:
        cv.waitKey(0)
        cv.destroyAllWindows()

def standardize_img(img, to=500):
    return cv.resize(img, (to, to))

def resize_faces(show_images=False):
    for face_letter in ['A', 'B', 'C']:
        for img_num in range(10):
            face = cv.imread(f'assets/faces/face{face_letter}{img_num}.jpg')
            std_face = standardize_img(face)
            cv.imwrite(f'assets/standardized/face{face_letter}{img_num}.jpg', std_face)
            if show_images:
                cv.imshow(f'original {face_letter}{img_num}', face)
                cv.imshow(f'resized {face_letter}{img_num}', std_face)
    if show_images:
        cv.waitKey(0)
        cv.destroyAllWindows()

def get_avg_faces(show_avg=False):
    for face_letter in ['A', 'B', 'C']:
        avg = np.zeros((500, 500, 3)).astype('uint8')
        for img_num in range(1, 10):
            face = cv.imread(f'assets/standardized/face{face_letter}{img_num}.jpg')
            weight = 1 / (img_num + 1)
            avg = cv.addWeighted(avg, 1 - weight, face, weight, 0)

        cv.imwrite(f'assets/standardized/stdFace{face_letter}.jpg', avg)
        if show_avg:
            cv.imshow(f'average face {face_letter}', avg)
    if show_avg:
        cv.waitKey(0)
        cv.destroyAllWindows()

def test_faces():
    for face_letter in ['A', 'B', 'C']:
        test_img = cv.imread(f'assets/images/test{face_letter}.jpg')
        test_face = get_face(test_img)
        test_std = standardize_img(test_face)
        
        best_match = None # (letter, error)
        for other_face_letter in ['A', 'B', 'C']:
            avg_face = cv.imread(f'assets/standardized/stdFace{other_face_letter}.jpg')
            diff = cv.subtract(test_std, avg_face)
            error = np.sum(np.abs(diff))
            if best_match is None or error < best_match[1]:
                best_match = (other_face_letter, error)

        print(f'Best match was img {best_match[0]} with error {best_match[1]} [TEST {"PASSED" if best_match[0] == face_letter else "FAILED"}]')
        
##### MAIN #####
generate_faces()
resize_faces()
get_avg_faces()
test_faces()