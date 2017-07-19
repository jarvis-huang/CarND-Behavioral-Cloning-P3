from keras.models import load_model

model = load_model('best/model_lenet.h5')
yaml_string = model.to_yaml()

text_file = open("arch.yaml", "w")
text_file.write("%s" % yaml_string)
text_file.close()
