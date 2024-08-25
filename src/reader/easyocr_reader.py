from easyocr import Reader
import cv2

class EasyOCRReader():
  
  def __init__(self) -> None:
    self.use_gpu = True
  
  def cleanup_text(self, text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

  def process_image(self):
    langs = ["en"]
    print("[INFO] OCR'ing with the following languages: {}".format(langs))
    # load the input image from disk
    image = cv2.imread("/Users/anand/Documents/personal/ocr/ocr_detection/images/C181a.jpeg")
    # OCR the input image using EasyOCR
    print("[INFO] OCR'ing input image...")
    reader = Reader(langs, gpu=self.use_gpu)
    results = reader.readtext(image)
    return (image, results)
  
  def process_results(self, image, results):
    # loop over the results
    for (bbox, text, prob) in results:
      # display the OCR'd text and associated probability
      print("[INFO] {:.4f}: {}".format(prob, text))
      # unpack the bounding box
      (tl, tr, br, bl) = bbox
      tl = (int(tl[0]), int(tl[1]))
      tr = (int(tr[0]), int(tr[1]))
      br = (int(br[0]), int(br[1]))
      bl = (int(bl[0]), int(bl[1]))
      # cleanup the text and draw the box surrounding the text along
      # with the OCR'd text itself
      text = self.cleanup_text(text)
      cv2.rectangle(image, tl, br, (0, 255, 0), 2)
      cv2.putText(image, text, (tl[0], tl[1] - 10),
      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
  reader = EasyOCRReader()
  image, results = reader.process_image()
  reader.process_results(image, results)