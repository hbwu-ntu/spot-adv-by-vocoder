import scipy
import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile

class AudioSpec():
	def __init__(self, sr = 16000, num_m = 80, hop = 200, win = 800,
				fmin = 80, fmax = 7600, power = 1.5, gl_iters = 100):
		self.sr = sr
		self.hop = hop
		self.win = win
		self.power = power
		self.gl_iters = gl_iters
		self.mb = librosa.filters.mel(sr, win, n_mels = num_m, fmin = fmin, fmax = fmax)
		self.inv_mb = np.linalg.pinv(self.mb)

	def load_wav(self, path):
		sr, x = wavfile.read(path)
		x = x.astype(np.float32)
		x = x/np.max(np.abs(x))
		return x

	def save_wav(self, x, path):
		x *= 32767 / max(0.01, np.max(np.abs(x)))
		wavfile.write(path, self.sr, x.astype(np.int16))

	def lin(self, x):
		return np.abs(self._stft(x))

	def inv_lin(self, y):
		return self.gl(y**self.power)

	def mel(self, y):
		return np.dot(self.mb, y)
	
	def inv_mel(self, y):
		y = np.dot(self.inv_mb, y)
		y = np.maximum(1e-10, y)
		return self.gl(y**self.power)

	def gl(self, y):
		angles = np.exp(2j*np.pi*np.random.rand(*y.shape))
		y_complex = np.abs(y).astype(np.complex)
		y = self._istft(y_complex * angles)
		for i in range(self.gl_iters):
			angles = np.exp(1j * np.angle(self._stft(y)))
			y = self._istft(y_complex * angles)
		return y
	
	def _stft(self, x):
		return librosa.stft(x, n_fft = self.win, hop_length = self.hop, win_length = self.win)

	def _istft(self, y):
		return librosa.istft(y, hop_length = self.hop, win_length = self.win)

