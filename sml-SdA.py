from pandas import DataFrame as df
import numpy as np
from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers import AutoEncoder, Dense, Dropout

def prepareData():
	X =df.from_csv('dataset/uci-sml/X.csv')
	y =df.from_csv('dataset/uci-sml/y.csv')
	X2 =df.from_csv('dataset/uci-sml/X_test.csv')
	y2 =df.from_csv('dataset/uci-sml/y_test.csv')

	floatlist =['18:Meteo_Exterior_Piranometro', '9:Humedad_Habitacion_Sensor', '7:CO2_Habitacion_Sensor']
	boollist =['12:Precipitacion']
	yfloatlist =['4:Temperature_Habitacion_Sensor']

	X[floatlist] =X[floatlist].astype(float)
	X2[floatlist] =X2[floatlist].astype(float)
	X[boollist] =X[boollist].astype(bool)
	X2[boollist] =X2[boollist].astype(bool)
	y[yfloatlist] =y[yfloatlist].astype(float)
	y2[yfloatlist] =y2[yfloatlist].astype(float)

	X_train =X[floatlist+boollist+['Month', 'Hour']].values
	y_train =y.values
	X_test =X2[floatlist+boollist+['Month', 'Hour']].values[:270]
	y_test =y2.values[:270]

	return (X_train, X_test, y_train, y_test)


class SdA:
	class sdaModel:
		def __init__(self):
			self.name =None
			self.model =None
			self.score =None
			self.score_train =None


	def __init__(self):
		from os import mkdir
		from datetime import datetime

		self.batch_size = None
		self.nb_epoch = 100
		self.nb_hidden_layers = [1536, 1024, 512]
		self.nb_noise_layers = [0.6, 0.4, 0.2]
		self.lossMode ='mape'
		self.startTime =str(datetime.now().strftime("%Y%m%d-%H%M"))
		self.folder ="result/uci-sml/%s"%self.startTime
		self.modelFolder =self.folder+"/models"
		self.resultsFolder =self.folder+"/results"
		self.rsScore =[]
		mkdir(self.folder)
		mkdir(self.modelFolder)
		mkdir(self.resultsFolder)


	def prepare(self, Xshape, yshape):
		self.nb_hidden_layers.insert(0,Xshape[1])
		print(yshape)
		if len(yshape)==1: self.output_dim =1
		elif len(yshape)==2: self.output_dim =yshape[1]


	def training(self, X_train, X_test, y_train, y_test):

		if self.batch_size ==None: self.batch_size =0.3*X_train.shape[0]
		self.prepare(X_train.shape, y_train.shape)
		X_train_tmp = np.copy(X_train)
		nb_epoch = self.nb_epoch
		batch_size = self.batch_size
		nb_noise_layers = self.nb_noise_layers
		nb_hidden_layers = self.nb_hidden_layers
		output_dim =self.output_dim
		trained_encoders = []

		if len(nb_noise_layers) !=len(nb_hidden_layers)-1:
			raise Exception(
				'Noise Layers Error', 'Noise layer length is not correct. Hidden:%s Noise:%s'%(
					len(nb_hidden_layers), len(nb_noise_layers)
				)
			)

		else:
			for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
				print('Pre-training the layer: Input {} -> Output {}'.format(n_in, n_out))
				encoder =Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid')
				decoder =Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')

				ae = Sequential()
				#ae.add(GaussianNoise(nb_noise_layers[i-1], input_shape=(n_in, )))
				ae.add(Dropout(nb_noise_layers[i-1], input_shape=(n_in, )))
				ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
				ae.compile(loss='mape', optimizer='rmsprop')
				ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
				trained_encoders.append(ae.layers[1].encoder)
				X_train_tmp = ae.predict(X_train_tmp)


			print("Fine Tuning")
			model = Sequential()

			for encoder in trained_encoders:
				model.add(encoder)

			model.add(Dense(input_dim=nb_hidden_layers[-1], output_dim=output_dim, activation='linear'))
			model.compile(loss=self.lossMode, optimizer='rmsprop')
			model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test))
			self.sdaModel.name ="%sh_%sepoch_%noise_%sbatch_%s.csv"%(
				nb_hidden_layers[1], self.nb_epoch, self.nb_noise_layers[0], self.batch_size, score[:7]
			)
			self.sdaModel.model =model
			self.sdaModel.score =model.evaluate(X_test, y_test)
			self.sdaModel.score_train =model.evaluate(X_test, y_test)
			print("Score: ", score)

			return self.sdaModel


	def predicting(self, X):
		return self.sdaModel.model.predict(X)


	def evaluating(self, x, y):
		return self.sdaModel.model.evaluate(x, y)


	def gridSearchSdA(self, **kwargs): #kwargs: noise_layers, epoch, hidden_layers
		if 'epoch' not in kwargs: epoch =[i for i in range(10, 130, 20)]
		if 'hidden_layers' not in kwargs: hidden_layers =[[i-128, i, i+128] for i in range(512, 2049, 512)]
		if 'noise_layers' not in kwargs: noise_layers =0.6
		rsScore =[]
		sda =SdA()

		for h in hidden_layers:
			for e in epoch:
				for n in noise_layers:
					#[h.insert(0, X_train.shape[1]) for h in hidden_layer]
					sda.nb_epoch =e
					sda.nb_noise_layers =n
					sda.nb_hidden_layers =h
					test_score =sda.training(X_train, X_test, y_train, y_test)
					train_score =self.evaluating(X_train, X_test)
					sda.saveModel()
					sda.saveRs(pred, y_test, score)

		sda.saveRsTable()
		print(sda.folder)
		print(sda.rsScore)


	def loadModel(self, architecture, weights):
		from keras.models import model_from_json

		modelname =architecture.split('/')[-1]
		model =model_from_json(open(architecture).read())
		model.load_weights(weights)
		self.sdaModel.name =modelname
		self.sdaModel.model =model
		print("Load Model: ", modelname)


	def saveModel(self):
		open("%s/%s_architecture.json"%(self.modelFolder, self.sdaModel.name), 'w').write(self.sdaModel.model.to_json())
		self.sdaModel.model.save_weights('%s/%s_weights.h5'%(self.modelFolder, self.sdaModel.name))


	def saveRs(self, pred, y_test, score):
		self.rsScore.append({"batch": self.batch_size, "epoch": self.nb_epoch, "score": score})
		df([[i[0], k[0]] for i, k in zip(pred, y_test)]).to_csv(
			"%s/%s"%(self.resultsFolder, self.sdaModel.name), index= False, index_label=False
		)


	def saveRsTable(self):
		f =open(self.folder+"/resultsTable.json", 'w')
		f.write(str(self.rsScore))
		f.close()



if __name__ =="__main__":
	rsScore =[]
	epoch =[i for i in range(10, 50, 10)]
	hidden_layer =[[i-128, i, i+128] for i in range(512, 2049, 512)]

	X_train, X_test, y_train, y_test =prepareData()


	sda =SdA()
	#[h.insert(0, X_train.shape[1]) for h in hidden_layer]

	for e in epoch:
		for h in hidden_layer:
			sda.batch_size =b
			sda.nb_epoch =e
			sda.nb_hidden_layers =h
			score =sda.training(X_train, X_test, y_train, y_test)
			pred =sda.predicting(X_test)
			sda.saveModel()
			sda.saveRs(pred, y_test, score)

	sda.saveRsTable()
	print(sda.folder)
	print(sda.rsScore)