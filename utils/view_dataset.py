import os
import argparse
from PIL import Image

print(os.getcwd())

if __name__== '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cub', help='dataset name')
	parser.add_argument('--text_path', type=str, default=os.path.join(os.getcwd(), 'cub_icml_txt'), help='path to text folder')
	parser.add_argument('--image_path', type=str, default=os.path.join(os.getcwd(),'CUB_200_2011/CUB_200_2011/images'), help='path to image folder')
	parser.add_argument('--text_num', type=int, default=5, help='number of texts to present')

	args = parser.parse_args()

	dataset = args.dataset
	text_path = args.text_path
	image_path = args.image_path
	text_num = args.text_num

	html = '<html><body><h1>{} dataset</h1><table border="1px solid gray" style="width=100%"><tr><td><b>File</b></td><td><b>Caption</b></td><td><b>Image</b></td></tr>'.format(dataset)

	for c in os.listdir(text_path):
		text_cls = os.path.join(text_path, c)
		image_cls = os.path.join(image_path, c)
		if not os.path.isdir(text_cls):
			continue
		for t in os.listdir(text_cls):
			text_file = os.path.join(text_cls, t)
			image_file = os.path.join(image_cls, t[:-4] + ".jpg")
			#print(image_file)

			fo = open(text_file, "r")
			txt_list = [raw_t.strip() for raw_t in fo.read().split(".")]
			txt = "<br>".join(txt_list[:text_num])
			#print(txt)
			fo.close()

			#img = Image.open(image_file)
			#img.save(os.path.join(dir_name, t[:-4] + ".jpg"))

			html = html + '\n<tr><td>{}</td><td>{}</td><td><img src="{}" width="120" height="120"/></td></tr>'.format(t[:-4], txt, image_file)


	html = html + "</html>"

	result = '{}.html'.format(dataset)
	html_file = open(result, "w")
	html_file.write(html)
	html_file.close()

