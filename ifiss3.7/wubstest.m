%WUBSTEST generate label data without using parallel toolbox
%   IFISS scriptfile: 26 December 2025
% Copyright (c) 2024 D.J. Silvester
global BATCH bsn FID
setpath
BATCH=1;
DELTA=0;
tstart = tic; %------- start timing
fprintf('Generating Wubs problem test data set ... ')

%----- setup problem data files
copyfile('boussinesq_flow/test_problems/lateral_bc.m', 'diffusion/specific_bc.m');
copyfile('stokes_flow/test_problems/no_flow.m', 'stokes_flow/specific_flow.m');
copyfile('diffusion/test_problems/zero_bc.m', 'stokes_flow/stream_bc.m');

% ---- preprocessing to set up grid and coefficient matrices
testproblem=['B-NS42_grid_batch'];
%---------- write the batch file
      [KID,message]=fopen(testproblem,'w');
      fprintf(KID,'1%%\n0.051%%\n2%%\n32%%\n8%%\n2%%\n1.5%%\n1%%\n1%%\n');
      fprintf(KID,'%59s\n','%---------- grid data file for Wubs test problem');
      fclose(KID);

%---------- set up matrices
      [FID,message]=fopen(testproblem,'r');
      if strcmp(message,'')~=1, error(['INPUT FILE ERROR: ' message])
      else, disp(['Working in batch mode from data file ',testproblem])
      end
      bsn=2; cavity_boussX

%------- run through test data points in parallel

rayleig = [2.87e9, 2.89e9];  %<--- Rayleigh numbers
prandtl = [1000,1000];   %<--- Prandtl numbers

nra=length(rayleig); npr=length(prandtl);
if nra~=npr, error('Oops ... check data setup!'), end

%---------- parallel matlab loop
for test=1:nra
      tt=test;
      fprintf(['\n [%g]'],tt)
      ra=rayleig(tt); RA=ra;
      fprintf(['\n Rayleigh number %g'],ra)
      pr=prandtl(tt); 
      fprintf(['\n Prandtl number is %g\n'],pr)
      testproblem=['B-NS42_test',num2str(tt)];
      batchfile=[testproblem,'_batch.m'];
%      gohome, cd batchfiles
%---------- write the batch file
      [FID,message]=fopen(batchfile,'w');
      fprintf(FID,'%g%%\n%g%%\n',ra,pr);
      fprintf(FID,'0%%\n1e13%%\n10%%\n2e-5%%\n2%%\n30%%\n0%%\n0%%\n13%%\n');
      fprintf(FID,'%53s\n','%---------- data file for for Wubs test problem');
      fclose(FID);
%---------- execute the batch file and compute the label
      batchprocess(testproblem)
end
etoc = toc(tstart);
fprintf('Bingo!\n')
fprintf('\n  %9.4e seconds\n\n\n',etoc)

%---------- open results file
[RID,message]=fopen('wubs_labels.txt','w');
fprintf(RID,'%33s\n','%------------ Wubs stretched grid32 label data');

for test=1:nra
   testresults=['Bouss_output',num2str(test),'.mat'];
   load(testresults)
   kk=KE.*KE*0.051;
%---------- check for incomplete time integration
   if 2001==length(time), flag=0;
   figure(100+test),subplot(121)
   plot([120:2000],kk(120:2000),'.-r'), axis square
   xlabel('step')
   subplot(122)
   plot([1200:2000],kk(1200:2000),'.-r'), axis square
   xlabel('step')
%---------- check for oscillating solution
   meanE=mean(kk(1200:2000));
   maxE=max(kk(1200:2000));
   minE=min(kk(1200:2000));
   testE=(maxE-minE)/meanE;
      if testE<5e-4, label=0;
      else, label=1; end
   else
%------------- steady state reached before 2000 steps
   flag=1;
   meanE = kk(end); testE=0; label=0;
   end
%---------- write the result to the file
   fprintf(RID,'%g,%g,%g,%g,%g,%g\n',Ra,Pr,label,meanE,testE,flag);
   fprintf('Label results saved for test %g\n',test)
end
fclose(RID);
fprintf('All done\n')

