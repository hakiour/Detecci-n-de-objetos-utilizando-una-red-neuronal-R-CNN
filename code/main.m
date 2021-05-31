clearvars,
close all,
clc,

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
% PROYECTO PSIV: Detección de objetos utilizando una red neuronal R-CNN
% 	-Carla Viñas Templado - 1564580
%   -Hamza Akiour - 1567215
global DESCARGAR_FOTOS;
global TRAINING_IMAGES_FOLDER;
global MOSTRAR_IMAGENES_MUESTRA;
global NUM_IMAGENES_MUESTRA;
global URL_DATASET;
global REALIZAR_ENTRENAMIENTO_DATASET;
global REALIZAR_ENTRENAMIENTO_OBJETO;
global VISUALIZAR_IMAGENES_PREVIEW;


DESCARGAR_FOTOS = 0;
MOSTRAR_IMAGENES_MUESTRA = 1;
VISUALIZAR_IMAGENES_PREVIEW = 1;
NUM_IMAGENES_MUESTRA = 36;
TRAINING_IMAGES_FOLDER = "cifar";
URL_DATASET = "https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz";
REALIZAR_ENTRENAMIENTO_DATASET = 0;
REALIZAR_ENTRENAMIENTO_OBJETO = 0;

%% Carga de datos
if DESCARGAR_FOTOS == 1
    %descargamos las fotos del training set (50 mil fotos... jeje)
    unpackedData = fullfile(TRAINING_IMAGES_FOLDER, 'dataset_' + TRAINING_IMAGES_FOLDER);
    if ~exist(unpackedData, 'dir')
        disp('Descargando dateset...');     
        untar(URL_DATASET, 'dataset_' + TRAINING_IMAGES_FOLDER); 
        disp('Dataset descargado.\n');
    end
else
    disp('Omitiendo descarga dataset');
end

%cargamos las fotos del training set
[trainingImages, trainingLabels, testImages, testLabels] = helperCIFAR10Data.load(TRAINING_IMAGES_FOLDER);
disp(size(trainingImages,4) +" imágenes training con un tamaño de " + size(trainingImages,1) + "x" + size(trainingImages,2) + " cargadas")

%Categorias
numImageCategories = size(categories(trainingLabels),1);
disp(numImageCategories + " categorias cargadas.");

if MOSTRAR_IMAGENES_MUESTRA == 1
    figure
    thumbnails = trainingImages(:,:,:,1:NUM_IMAGENES_MUESTRA);
    montage(thumbnails)
end

%% Creación de la red CNN
[height, width, numCanales, fotos] = size(trainingImages); %Lo creamos en base al tamaño de las imagenes

imageSize = [height width numCanales];
filtros_size = [5 5];
filtros = 32;

%Creamos la capa inicial

capas= [
    %Creamos una capa (la inicial) con el size del tamaño de nuestras
    %imagenes
    imageInputLayer(imageSize);
    
    %CAPA INTERMEDIA
    %En esta capa añadimos un padding de 2 para no perder información con
    %los brodes
    convolution2dLayer(filtros_size,filtros,'Padding',2)
    %ponemos una capa relu
    reluLayer()
    %ponemos una capa de pooling, (3x3 spatial pooling area con un stride de 2
    %pixeles). Esto también nos hace pasar en nuestro caso de imagenes de 32*32
    %a 15*15
    maxPooling2dLayer(3,'Stride',2)
    
    % Repeat the 3 core capas to complete the middle of the network.
    convolution2dLayer(filtros_size,filtros,'Padding',2)
    reluLayer()
    maxPooling2dLayer(3, 'Stride',2)
    
    convolution2dLayer(filtros_size,2 * filtros,'Padding',2)
    reluLayer()
    maxPooling2dLayer(3,'Stride',2)
    
    %%CAPA FINAL
    %Ponemos una capa con 64 neuronas
    fullyConnectedLayer(64)
    reluLayer
    %Añadimos una capa final totalmente conectada
    %Obtenemos las señales
    fullyConnectedLayer(numImageCategories)
    %metmos la capa softmax, la cual es exponencial
    softmaxLayer
    %metemos la capa de clasificación
    classificationLayer
];

%Inicializamos la primera convlución con pesos aleatorios utilizando una
%distribución normal
capas(2).Weights = 0.0001 * randn([filtros_size numCanales filtros]);

% Parametros de entrenamiento
opciones = trainingOptions('sgdm','Momentum', 0.9,'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise','LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, 'L2Regularization', 0.004, 'MaxEpochs', 40, ...
    'MiniBatchSize', 128, 'Verbose', true);

if REALIZAR_ENTRENAMIENTO_DATASET == 1    
    cifar10Net = trainNetwork(trainingImages, trainingLabels, capas, opciones);
else
    % Cargamos el set predefinido
    load('rcnnStopSigns.mat','cifar10Net')       
end

% Pesos de la primera capa convolucional
w = cifar10Net.Layers(2).Weights;
% Escalamos para que se vea correctamente
w = rescale(w);
if VISUALIZAR_IMAGENES_PREVIEW == 1
    figure
    montage(w)
end

% Ejecutamos la red en el set de prueba
test_prueba = classify(cifar10Net, testImages);

% Calculanos la precision.
precision = sum(test_prueba == testLabels)/numel(testLabels);
disp("Precision de la red: " + precision)

%% Cargamos set de datos del objeto
data = load('stopSignsAndCars.mat', 'stopSignsAndCars');
stopSignsAndCars = data.stopSignsAndCars;

%  Ponemos el path
visiondata = fullfile(toolboxdir('vision'),'visiondata');
stopSignsAndCars.imageFilename = fullfile(visiondata, stopSignsAndCars.imageFilename);

disp("Imágenes cargadas objeto 1: " + size(stopSignsAndCars,1))

% Nos quedamos solo con las etiquetas de stopSing
stopSigns = stopSignsAndCars(:, {'imageFilename','stopSign'});

%% Entrenamiento objeto 
if REALIZAR_ENTRENAMIENTO_OBJETO
    
    % Set training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    % Train an R-CNN object detector. This will take several minutes.    
    rcnn = trainRCNNObjectDetector(stopSigns, cifar10Net, options, ...
    'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange',[0.5 1])
else
    % Load pre-trained network for the example.
    load('rcnnStopSigns.mat','rcnn')       
end
 
%imagen para verificar que se identifica la foto
testImage = imread('stopSignTest.jpg');

% Detección de las señales de stop
[bboxes,score,label] = detect(rcnn,testImage,'MiniBatchSize',128)


% Vemos los resultados
[score, idx] = max(score);

bbox = bboxes(idx, :);
annotation = sprintf('%s: (Precisión de %f)', label(idx), score);

outputImage = insertObjectAnnotation(testImage, 'rectangle', bbox, annotation);

figure
imshow(outputImage)