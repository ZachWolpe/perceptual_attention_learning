Fs = 44100;
T = 1.5;
t = linspace(0,T,Fs*T);
f_low = 220;
f_high = 3520;
n = 768;
step = (f_high/f_low)^(1/(n-1));

f = f_low;



Tfade = 0.2;
t_in = Fs*Tfade;
t_out = Fs*T - Fs*Tfade;

for i = 1:n
    w = 2*pi*f;
    s = sin(w*t);
    for j=1:t_in
        w_env = 2*pi*1/(Tfade * 4);
        s(1, j) = s(1,j)*sin(w_env*t(j));
    end 
    for j=t_out:(T*Fs)
        w_env = 2*pi*1/(Tfade * 4);
        %s(1, j) = s(1,j)*sin(w_env*t(j)+pi/2);
        s(1, j) = s(1,j)*sin(w_env*t(j-T*Fs+2*t_in));
    end
    %filename = append('sound_15_', int2str(i), '.wav');
    %filename = append('sound_', int2str(i), '_',int2str(f), '.wav');
    %audiowrite(filename, s, Fs);
    f = f * step;
    disp('i');
    disp(i);
    format shortG
    disp(f);
end

plot(t, s)


%{
white_noise = wgn(Fs,T,0);

for i=1:t_in
    w_env = 2*pi*T/(Tfade * 4);
    white_noise(i, 1) = white_noise(i,1)*sin(w_env*t(i));
   
end
for i=t_out:(T*Fs)
    w_env = 2*pi*T/(Tfade * 4);
    white_noise(i, 1) = white_noise(i, 1)*sin(w_env*t(i)+pi/2);
    
    
end
audiowrite('white_noise.wav', white_noise, Fs);

pink_noise = pinknoise(Fs);
for i=1:t_in
    w_env = 2*pi*T/(Tfade * 4);
    pink_noise(i, 1) = pink_noise(i,1)*sin(w_env*t(i));
   
end
for i=t_out:(T*Fs)
    w_env = 2*pi*T/(Tfade * 4);
    pink_noise(i, 1) = pink_noise(i, 1)*sin(w_env*t(i)+pi/2);
    
    
end
plot(t, pink_noise);
audiowrite('pink_noise.wav', pink_noise, Fs);

%}