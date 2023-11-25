from distutils.log import debug
from fileinput import filename
from flask import *
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import os
from werkzeug.utils import secure_filename

# So the colors we want to use for the design are:
# 1) Erin Green: 00FF40 (Matrix color)
# 2) Blue : 0066FF
# 3) Red: CA0027
# The blue and red are about 120 degrees off either way from Erin Green in LAB space.

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "gif", "png", "jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET','POST'])
def main():
    return render_template("index.html", img=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))


@app.route('/toolhub', methods=['POST'])
def toolhub():
    if request.method == 'POST':
        if "file" not in request.files:
            print('failed')
            return render_template("toolhub.html", name="failed", img="static/logo.png")
        f = request.files['file']
        print(f.filename[-3:])
        if f.filename[-3:] =="jpg" or f.filename[-4:]=="jpeg":
            f.save(os.path.join(app.config["UPLOAD_FOLDER"],"test.jpg"))
            im=cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg"))
            cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"),im)
        elif f and allowed_file(f.filename):
            f.save(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        else:
            return render_template("toolhub.html", name="failed", img="static/logo.png")

        return render_template("toolhub.html", name=f.filename, img=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))

@app.route('/lum_laplacian', methods=['POST'])
def lum_laplacian():
    if request.method == 'POST':
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), cv2.IMREAD_GRAYSCALE)
        lap_image = cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=3)
        lap_image = np.abs(lap_image)
        print(np.max(lap_image))
        print(np.min(lap_image))
        norm_bool = request.form.getlist("normalize_noise")  # [u'Item 1'] []
        norm_slider = request.form.getlist("normalize_scale")
        if norm_slider:
            norm_slider = int(norm_slider[0])
        else:
            norm_slider = 255
        if norm_bool:
            lap_image = cv2.normalize(src=lap_image, dst=lap_image, alpha=0, beta=float(norm_slider), norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_32F)
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "lap.png"), lap_image)
        return render_template("lum_laplacian.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "lap.png"), init_scale=norm_slider)

@app.route('/lum_grad', methods=['POST'])
def lum_grad(): #This doesn't "look good," go take effort to make the color maps look nice.
    if request.method == 'POST':
        eps = 7.0/3.0 - 4.0/3.0 - 1
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"),cv2.IMREAD_GRAYSCALE)
        sobel_x = cv2.Scharr(image, ddepth=cv2.CV_16S, dx=1, dy=0)
        sobel_y = cv2.Scharr(image, ddepth=cv2.CV_16S, dx=0, dy=1)
        lg = np.zeros([np.shape(image)[0],np.shape(image)[1],3])
        lg[:,:,1] = sobel_x
        lg[:,:,2] = sobel_y #You should map these to two independent color maps for visibility, instead of just different channels. Also, make sure the color bars are around.
        norm_bool = request.form.getlist("normalize_noise")  # [u'Item 1'] []
        norm_slider = request.form.getlist("normalize_scale")
        if norm_slider and norm_bool:
            norm_slider = int(norm_slider[0])
        else:
            norm_slider = 255
        lg = cv2.normalize(src=lg, dst=lg, alpha=0, beta=float(norm_slider), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lg[:,:,0] = np.zeros([np.shape(image)[0], np.shape(image)[1]])
        print(lg)
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "lg.png"),lg)
        return render_template("lum_grad.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "lg.png"), init_scale=norm_slider)

@app.route('/noise_gauss', methods=['POST'])
def noise_gauss():
    if request.method == 'POST':
        eps = 7.0/3.0 - 4.0/3.0 - 1

        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        blur_image = cv2.GaussianBlur(image, (5,5), 0)
        noise_image = np.abs(image - blur_image)
        norm_bool = request.form.getlist("normalize_noise")  # [u'Item 1'] []
        norm_slider = request.form.getlist("normalize_scale")
        if norm_slider:
            norm_slider = int(norm_slider[0])
        else:
            norm_slider = 255
        if norm_bool:
            noise_image = noise_image*float(norm_slider) / (np.max(noise_image) - np.min(noise_image) + eps)
            print(np.max(noise_image))
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "gauss_noise.png"), noise_image)
        return render_template("noise_gauss.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "gauss_noise.png"), init_scale=norm_slider)

@app.route('/noise_median', methods=['POST'])
def noise_median():
    if request.method == 'POST':
        eps = 7.0 / 3.0 - 4.0 / 3.0 - 1
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        blur_image = cv2.medianBlur(image, 5)
        noise_image = np.abs(image - blur_image)
        norm_bool = request.form.getlist("normalize_noise")  # [u'Item 1'] []
        norm_slider = request.form.getlist("normalize_scale")
        if norm_slider:
            norm_slider = int(norm_slider[0])
        else:
            norm_slider = 255
        if norm_bool:
            noise_image = noise_image*float(norm_slider) / (np.max(noise_image) - np.min(noise_image) + eps)
            print(np.max(noise_image))
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "sp_noise.png"), noise_image)
        return render_template("noise_median.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "sp_noise.png"), init_scale=norm_slider)

@app.route('/noise_bilateral', methods=['POST'])
def noise_bilateral():
    if request.method == 'POST':
        eps = 7.0 / 3.0 - 4.0 / 3.0 - 1
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        blur_image = cv2.bilateralFilter(image,5,70,70)
        noise_image = np.abs(image - blur_image)
        norm_bool = request.form.getlist("normalize_noise")  # [u'Item 1'] []
        norm_slider = request.form.getlist("normalize_scale")
        if norm_slider:
            norm_slider = int(norm_slider[0])
        else:
            norm_slider = 255
        if norm_bool:
            noise_image = noise_image*float(norm_slider) / (np.max(noise_image) - np.min(noise_image) + eps)
            print(np.max(noise_image))
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "compression_noise.png"), noise_image)
        return render_template("noise_bilateral.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "compression_noise.png"), init_scale=norm_slider)

@app.route('/ela', methods=['POST'])
def ela():
    if request.method == 'POST':
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        compress_slider = request.form.getlist("compress_scale")
        if compress_slider:
            compress_slider = int(compress_slider[0])
            print(compress_slider)
        else:
            compress_slider = 95
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "ela_compressed_image.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), compress_slider])
        compressed_image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "ela_compressed_image.jpg"))
        noise = np.abs(image-compressed_image)
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "ela_noise.png"), noise)
        return render_template("ela.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2=os.path.join(app.config["UPLOAD_FOLDER"], "ela_noise.png"), init_scale=compress_slider)

@app.route('/surf', methods=['POST'])
def surf():
    if request.method == 'POST':
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        lap_image = cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=3)
        cv2.imwrite("static/lap.png", lap_image)
        return render_template("surf.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2="static/lap.png")

@app.route('/ai_detectors', methods=['POST'])
def ai_detectors():
    if request.method == 'POST':
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        lap_image = cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=3)
        cv2.imwrite("static/lap.png", lap_image)
        return render_template("ai_detectors.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2="static/lap.png")

@app.route('/splicing_detectors', methods=['POST'])
def splicing_detectors():
    if request.method == 'POST':
        image = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))
        lap_image = cv2.Laplacian(image, ddepth=cv2.CV_16S, ksize=3)
        cv2.imwrite("static/lap.png", lap_image)
        return render_template("splicing_detectors.html", img1=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"), img2="static/lap.png")

@app.route('/metadata', methods=['POST'])
def metadata():
    if request.method == 'POST':
        image = Image.open(os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg"))
        exifdata = image.getexif()
        outstring = "Metadata: \n"
        for tag_id in exifdata:
            tag=TAGS.get(tag_id,tag_id)
            data=exifdata.get(tag_id)
            outstring = outstring + str(tag) + ": \t" + str(data) + "\n"
        print(outstring)
        outstring = outstring.split('\n')
        print(outstring)
        return render_template("metadata.html", img=os.path.join(app.config["UPLOAD_FOLDER"], "test.jpg"), words=outstring)

@app.route('/tutorial', methods=['POST'])
def tutorial():
    if request.method == 'POST':
        return render_template("tutorial.html", img=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))

@app.route('/contact', methods=['POST'])
def contact():
    if request.method == 'POST':
        return render_template("contact.html", img=os.path.join(app.config["UPLOAD_FOLDER"], "test.png"))


if __name__ == '__main__':
    app.run(debug=True)