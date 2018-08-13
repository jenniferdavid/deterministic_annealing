
H = load("hist");
iterations = H(1:end,1); 
max = H(1:end, 2);
X = [max,iterations];
hist3(X);
xlabel('iteration'); ylabel('max');
set(gcf,'renderer','opengl');
%%
% Color the bars based on the frequency of the observations, i.e. according
% to the height of the bars.
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');

