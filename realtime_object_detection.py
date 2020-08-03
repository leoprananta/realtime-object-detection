# import packages
# jika ada error ModuleNotFoundError: No module named 'imutils' dan 'cv2', maka instal melalui pip install opencv-python dan pip install imutils

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# argument construct atau parameter run command
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# inisialisasi daftar class labels MobileNetSSD yg sudah di training
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# deteksi, lalu generate color box atau bingkai utuk object
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load model dari folder
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# inisialisasi video stream, mengijinkan penggunaan kamera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# inisialisasi FPS counter
fps = FPS().start()

# looping per frames dari kamera
while True:
	# ambil bingkai dari video dan ubah ukurannya
	# setting maximum width 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# ambil dimensi bingkai lalu mengubah menjadi blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# lewatkan blob melalui jaringan untuk mendapatkan deteksi lalu
	# prediksi
	net.setInput(blob)
	detections = net.forward()

	# looping deteksi
	for i in np.arange(0, detections.shape[2]):
		# ekstrak keakuratan(var confidence) (misal probabilitas) yang terkait dengan
		# prediksi
		confidence = detections[0, 0, i, 2]

		# memfilter deteksi lemah dengan memastikan keakuratan `confidence`
		# lebih besar dari keakuratan minimum
		if confidence > args["confidence"]:
			# ekstrak indeks label kelas dari
			# `detections`, kemudian hitung-(x, y) -koordinat dari
			# kotak pembatas objek
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# gambarkan prediksi pada bingkai
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# tampilkan frame output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# kalau dipencet 'q', hentikan looping alias berhenti
	if key == ord("q"):
		break

	# update FPS counter
	fps.update()

# stop timer dan tampilkan info FPS
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# clear
cv2.destroyAllWindows()
vs.stop()

# Struktur perintah penggunaan = python fileName --prototxt filenName --model fileName
# python realtime_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
