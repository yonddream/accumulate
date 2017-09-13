function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


    % 假设原始样本中有两类，其中：
    % 1：总共有 P个类别为1的样本，假设类别1为正例。 
    % 2：总共有N个类别为0 的样本，假设类别0为负例。 
    % 经过分类后：
    % 3：有 TP个类别为1 的样本被系统正确判定为类别1，FN 个类别为1 的样本被系统误判定为类别 0，
    % 显然有P=TP+FN； 
    % 4：有 FP 个类别为0 的样本被系统误判断定为类别1，TN 个类别为0 的样本被系统正确判为类别 0，
    % 显然有N=FP+TN； 
     
    % 那么：
    % 精确度（Precision）：
    % P = TP/(TP+FP) ;  反映了被分类器判定的正例中真正的正例样本的比重（ 
     
    % 准确率（Accuracy）
    % A = (TP + TN)/(P+N) = (TP + TN)/(TP + FN + FP + TN);    
    % 反映了分类器统对整个样本的判定能力——能将正的判定为正，负的判定为负 
     
    % 召回率(Recall)，也称为 True Positive Rate:
    % R = TP/(TP+FN) = 1 - FN/T;  反映了被正确判定的正例占总的正例的比重 
     
    % 转移性（Specificity，不知道这个翻译对不对，这个指标用的也不多），
    % 也称为 True NegativeRate 
    % S = TN/(TN + FP) = 1 – FP/N；   明显的这个和召回率是对应的指标，
    % 只是用它在衡量类别0 的判定能力。 
     
    % F-measure or balanced F-score
    % F = 2 *  召回率 *  准确率/ (召回率+准确率)；这就是传统上通常说的F1 measure，

    pred = (pval < epsilon);
    
    tp = sum((pred == 1) & (yval == 1));
    fp = sum((pred == 1) & (yval == 0));
    fn = sum((pred == 0) & (yval == 1));
    
    prec = tp/(tp + fp);
    rec = tp/(tp + fn);
    F1 = (2 * prec * rec) /(prec + rec);



    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
