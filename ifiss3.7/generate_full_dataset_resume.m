% generate_full_dataset_resume.m
% Генерация полного датасета с возможностью продолжения после остановки
% Исправлена запись в файл с абсолютным путём

clear; clc;
global BATCH bsn FID RA DELTA

% --- Параметры сетки параметров ---
ras = 1300:5:1600;                    % 61 значение Rayleigh
deltaset = [1e-16, 1e-12, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2]; % 7 значений
nra = length(ras);
ndelta = length(deltaset);
total_points = nra * ndelta;

% --- Параметры интегрирования ---
target_time = 1e16;
step_multiplier = 4;      % 4 * 200 = 800 шагов
accuracy_tol = 1e-6;
avg_freq = 30;

% --- Подготовка файлов граничных условий (один раз) ---
fprintf('Копирование файлов граничных условий...\n');
copyfile('boussinesq_flow/test_problems/bottom_bcX.m', 'diffusion/specific_bc.m');
copyfile('stokes_flow/test_problems/no_flow.m', 'stokes_flow/specific_flow.m');
copyfile('diffusion/test_problems/zero_bc.m', 'stokes_flow/stream_bc.m');

% --- Проверка наличия файлов сетки ---
gohome
if ~exist('datafiles','dir')
    mkdir('datafiles');
end
gridFile = fullfile('datafiles', 'rect_grid1h.mat');
boussFile = fullfile('datafiles', 'rect_bouss_nobc.mat');
if ~exist(gridFile, 'file') || ~exist(boussFile, 'file')
    error('Файлы сетки не найдены. Сначала запустите generate_ray_data для создания сетки.');
end
fprintf('Файлы сетки найдены. Загружаем...\n');
cd datafiles;
load rect_grid1h.mat;
load rect_bouss_nobc.mat;
gohome;

% --- Создаём/открываем файл результатов с абсолютным путём ---
resultDir = fullfile(pwd, 'datafiles');
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end
resultFile = fullfile(resultDir, 'full_labels.txt');
if ~exist(resultFile, 'file')
    fid = fopen(resultFile, 'w');
    if fid == -1
        error('Не удалось создать файл %s', resultFile);
    end
    fprintf(fid, '%%------------ Rayleigh Benard grid32 full label results\n');
    fprintf(fid, '%% Ra, delta, label, KE(end), VTY(end), flag\n');
    fclose(fid);
end

% --- Читаем уже выполненные комбинации ---
fprintf('Проверка уже выполненных расчётов...\n');
done = [];
if exist(resultFile, 'file')
    data = readlines(resultFile);
    data = data(~startsWith(data, '%'));   % убираем строки с комментариями
    for k = 1:length(data)
        parts = strsplit(data{k}, ',');
        if length(parts) >= 2
            ra_done = str2double(parts{1});
            delta_done = str2double(parts{2});
            done = [done; ra_done, delta_done];
        end
    end
end
fprintf('Найдено %d уже выполненных комбинаций.\n', size(done,1));

% --- Основной цикл ---
tstart = tic;
processed = 0;
for i = 1:nra
    for j = 1:ndelta
        ra = ras(i);
        delta = deltaset(j);
        
        % Пропускаем, если уже есть
        if ~isempty(done) && any(done(:,1)==ra & done(:,2)==delta)
            continue;
        end
        
        processed = processed + 1;
        fprintf('\n--- Расчёт %d из %d: Ra=%g, delta=%g ---\n', ...
            processed, total_points - size(done,1), ra, delta);
        
        % Устанавливаем глобальные переменные для batchprocess
        RA = ra;
        DELTA = delta;
        
        % Уникальное имя для временных файлов
        testname = sprintf('B-NS42_test_%d_%d', i, j);
        batchfile = [testname, '_batch.m'];
        
        % Создаём batch-файл
        fid = fopen(batchfile, 'w');
        fprintf(fid, '%g%%\n7.1%%\n%g%%\n', ra, delta);
        fprintf(fid, '%g%%\n%d%%\n%g%%\n0%%\n%d%%\n0%%\n0%%\n13%%\n', ...
            target_time, step_multiplier, accuracy_tol, avg_freq);
        fprintf(fid, '%53s\n', '%---------- data file');
        fclose(fid);
        
        % Запускаем расчёт
        batchprocess(testname);
        
        % Ожидаемое имя файла результатов (batchprocess создаёт Bouss_output_i_j.mat)
        resfile = sprintf('Bouss_output_%d_%d.mat', i, j);
        if exist(resfile, 'file')
            data = load(resfile, 'KE', 'VTY', 'time');
            KE = data.KE;
            VTY = data.VTY;
            time = data.time;
            ke_final = KE(end);
            omega_final = VTY(end);
            label = double(ke_final >= 3e-3);
            flag = double(length(time) ~= 801);
            
            % --- Запись результата ---
            % Закрываем все открытые файлы
            fclose('all');
            % Пытаемся открыть файл
            fid_out = fopen(resultFile, 'a');
            if fid_out == -1
                % Если не открывается, создаём резервный файл
                backupFile = fullfile(resultDir, 'full_labels_backup.txt');
                fid_out = fopen(backupFile, 'a');
                if fid_out == -1
                    error('Не удалось открыть файл %s или %s для записи', resultFile, backupFile);
                else
                    fprintf('Предупреждение: использован резервный файл %s\n', backupFile);
                    resultFile = backupFile;
                end
            end
            fprintf(fid_out, '%g,%g,%d,%g,%g,%d\n', ra, delta, label, ke_final, omega_final, flag);
            fclose(fid_out);
            fprintf('   Результат: label=%d, KE=%g, VTY=%g\n', label, ke_final, omega_final);
            
            % Удаляем временные файлы
            delete(resfile);
            delete([testname, '.mat']);
            delete(batchfile);
        else
            warning('Не найден файл результатов %s для Ra=%g, delta=%g', resfile, ra, delta);
        end
    end
end

elapsed = toc(tstart);
fprintf('\n========================================\n');
fprintf('Все расчёты завершены!\n');
fprintf('Общее время: %.2f секунд (%.2f часов)\n', elapsed, elapsed/3600);
fprintf('Результаты сохранены в %s\n', resultFile);