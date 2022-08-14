import pyaudio
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import wave, struct



#GENRE	BPM
#Hip Hop	85–95 BPM
#Glitch Hop	105–115 BPM
#Techno	120–125 BPM
#House	115–130 BPM
#Electro	128 BPM
#Dubstep	140 BPM (with a half time, 70 BPM feel)
#Drum and Bass	174 BPM

if len(sys.argv)!=2:
	print("%s wavfile" % sys.argv[0])
	sys.exit(1)

wave_fn=sys.argv[1]


wavefile = wave.open(wave_fn, 'r')
print("Frame rate", wavefile.getframerate())
print("Sample width", wavefile.getsampwidth())
print("N-channels", wavefile.getnchannels())

chunk=4096
length = wavefile.getnframes()
sample_width=wavefile.getsampwidth()
frame_rate=wavefile.getframerate()
nchannels=wavefile.getnchannels()
f_vec = frame_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate

# mic sensitivity correction and bit conversion
mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor
mic_low_freq = 100 # low frequency response of the mic (mine in this case is 100 Hz)

#max beat
max_bpm=200
min_bpm=60

max_bps=max_bpm/60
min_bps=min_bpm/60

#prepare output
audio = pyaudio.PyAudio()
stream_out = audio.open(
        format=audio.get_format_from_width(sample_width),
        channels=nchannels,
        rate=frame_rate, input=False, output=True)
stream_out.start_stream()

bytes_per_sample=sample_width*8

bass_low_freq_cutoff=100
bass_high_freq_cutoff=150
low_freq_loc = np.argmin(np.abs(f_vec-bass_low_freq_cutoff))
high_freq_loc = np.argmin(np.abs(f_vec-bass_high_freq_cutoff))

def analyze(wavedata_np):
	wavedata_np = ((wavedata_np/np.power(2.0,15))*5.25)*(mic_sens_corr) 
	fft_data = (np.abs(np.fft.fft(wavedata_np))[0:int(np.floor(chunk/2))])**2
	return fft_data[low_freq_loc:high_freq_loc] #.mean()
	# compute FFT parameters
	max_loc = np.argmax(fft_data[low_freq_loc:high_freq_loc])+low_freq_loc

	print("%0.3f @ %0.3e" % (f_vec[max_loc],fft_data[max_loc]))
	#return
	# plot
	plt.style.use('ggplot')
	plt.rcParams['font.size']=18
	fig = plt.figure(figsize=(13,8))
	ax = fig.add_subplot(111)
	plt.plot(f_vec,fft_data)
	ax.set_ylim([0,2*np.max(fft_data)])
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Amplitude [Pa]')
	ax.set_xscale('log')
	plt.grid(True)

	# max frequency resolution 
	plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (frame_rate/(2*chunk)),xy=(0.7,0.92),\
		     xycoords='figure fraction')

	# annotate peak frequency
	annot = ax.annotate('Freq: %2.1f'%(f_vec[max_loc]),xy=(f_vec[max_loc],fft_data[max_loc]),\
			    xycoords='data',xytext=(0,30),textcoords='offset points',\
			    arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')
	    
	#plt.savefig('fft_1kHz_signal.png',dpi=300,facecolor='#FCFCFC')
	plt.show()
	sys.exit(1)

n_mini_chunks=8
mini_chunk=chunk//n_mini_chunks
mini_chunks=[]
outputs=[]
for i in range(0, int(length/mini_chunk)):
    wavedata = wavefile.readframes(mini_chunk)
    wavedata_np = np.frombuffer(wavedata, dtype=f'int{bytes_per_sample}') #channels are interleaved
    mini_chunks.append(wavedata_np)
    if len(mini_chunks)>n_mini_chunks:
      mini_chunks.pop(0)
    if len(mini_chunks)==n_mini_chunks: 
      outputs.append(analyze(np.hstack(mini_chunks))) # offload to a buffer? multiprocess
    stream_out.write(wavedata)
    #time.sleep(0.2)
stream_out.stop_stream()
stream_out.close()
audio.terminate() 

outputs=np.vstack(outputs)

time_delta=(1/frame_rate)
mini_chunk_delta=mini_chunk*time_delta
times=np.arange(outputs.shape[0])*mini_chunk_delta

for idx in range(outputs.shape[1]):
	plt.plot(times,outputs[:,idx]/(outputs[:,idx].sum()),label="%d" % idx)
plt.yscale('log')
plt.ylim([1e-3, 1e-2])
plt.legend()
plt.show()

#fft on the power spectrum on lower bass sounds
outputs-outputs.mean(axis=0,keepdims=True)
half=int(np.floor(outputs.shape[0]/2))
f_vec_beats = (1/mini_chunk_delta)*np.arange(half)/len(outputs) # frequency vector based on window size and sample rate

for idx in range(outputs.shape[1]):
	fft_data = (np.abs(np.fft.fft(outputs[:,idx]))[0:half])**2
	plt.plot(f_vec_beats,fft_data,label="%d" % idx)
plt.legend()
plt.show()
valid_idxs=np.where(np.logical_and(f_vec_beats<max_bps,f_vec_beats>min_bps))
for idx in range(outputs.shape[1]):
	fft_data = (np.abs(np.fft.fft(outputs[:,idx]))[0:half])**2
	plt.plot(f_vec_beats[valid_idxs],fft_data[valid_idxs]/fft_data[valid_idxs].sum(),label="%d" % idx)
plt.yscale('log')
plt.show()

breakpoint()
#form_1 = pyaudio.paInt16 # 16-bit resolution
#chans = 1 # 1 channel
#samp_rate = 44100 # 44.1kHz sampling rate
#chunk = 8192 # 2^12 samples for buffer
#dev_index = 2 # device index found by p.get_device_info_by_index(ii)


#audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
#stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    #input_device_index = dev_index,input = True, \
                    #frames_per_buffer=chunk)

# record data chunk 
#stream.start_stream()
#data = np.fromstring(stream.read(chunk),dtype=np.int16)
#stream.stop_stream()

# (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections

