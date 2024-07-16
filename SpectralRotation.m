%% script to spectrally rotate audio 

% what file do you want to load in
filename = "Audio.wav";

% load in audio clip
[Audio,fs] = audioread(filename);

% optionally observe how frequency spectrum changes with spectral rotation
AudioFFT = fft(Audio(:,1));

% Plot the frequency spectrum
t = 0:1/fs:1-1/fs;             % Time vector
N = length(Audio);                 % Number of points in the time series
f = (0:N-1)*(fs/N);            % Frequency vector


magnitude = abs(AudioFFT)/N;          % Magnitude of the FFT
single_sided_magnitude = magnitude(1:N/2+1);  % Single-sided spectrum
single_sided_magnitude(2:end-1) = 2*single_sided_magnitude(2:end-1);  % Adjust the amplitude

% Define the single-sided frequency vector
f_single_sided = f(1:N/2+1);

figure;
plot(f_single_sided, single_sided_magnitude);
hold on
title('Single-Sided Amplitude Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Define the frequency range to rotate (55-2000 Hz)
freq_range = [55 2000];

% Identify indices corresponding to the frequency range
freq_indices = (f >= freq_range(1)) & (f <= freq_range(2));

% look at only certain indices
freq_idx = find(freq_indices);

% Apply the flipping to both negative and positive frequencies
X_rotated = AudioFFT;
X_rotated(freq_idx) = AudioFFT(flip(freq_idx)); % First half (positive frequencies)
X_rotated(N+1-freq_idx) = conj(AudioFFT(flip(N+1-freq_idx))); % Second half (negative frequencies)

% Reconstruct the signal using the inverse FFT
x_rotated = ifft(flip(X_rotated), 'symmetric');

% Plot the original and rotated signals
figure;
subplot(2, 1, 1);
plot(Audio);
title('Original Signal');
xlabel('Time (s)');
ylabel('Amplitude');

subplot(2, 1, 2);
plot(x_rotated(1:end-1));
title('Rotated Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Plot the original and rotated spectra
figure;
subplot(2, 1, 1);
plot(abs(AudioFFT)/N);
title('Original Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

subplot(2, 1, 2);
plot(abs(X_rotated)/N);
title('Rotated Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');

audiowrite("SpectralRotation.wav",x_rotated,fs)
