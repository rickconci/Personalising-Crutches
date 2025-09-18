function data = label_data(dataRaw)

time = dataRaw.data1 / 1000;
time = time - time(1);

data.Time = time;

data.IMUroll = dataRaw.data3;
data.IMUpitch = dataRaw.data4;
data.IMUyaw = dataRaw.data5;
data.IMUaccX = dataRaw.data6;
data.IMUaccY = dataRaw.data7;
data.IMUaccZ = dataRaw.data8;
data.IMUgyroX = dataRaw.data9;
data.IMUgyroY = dataRaw.data10;
data.IMUgyroZ = dataRaw.data11;

data.forceMeasured = dataRaw.data12;
end