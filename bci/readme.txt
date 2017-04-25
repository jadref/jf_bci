Set of functions to work with bio-signal datasets

Overview
--------

The basic idea of these functions is to initially construct an structure (conventionally denoted a 'z') which contains both the raw data (in z.X) as well as meta-data which allows interpertation of the structure.  Each of the functions (except the data-import functions) in this directory take such a structure as input, along with options as 'name',value pairs, and return the modifed structure as output.  This convention allows has significant advantages for rapid-prototyping of BCI experiments/analysis techniques:

* Pipeline based specfication of analysis methods; e.g. jf_cvtrain(jf_welchpsd(jf_detrend(jf_reref(z))))

* Self-documenting data structures -- the core datastructure as well as containing the raw data, contains meta-data describing it's structure (which dimensions are which), and the processing history of the object.  This history can be printed using the jf_disp(z) method.


In particular for each experiment the following meta-data is stored:

z.subj, z.expt, z.label, z.session -- this fields contain general information about the experiment/subject/session-for-this-subject and a label for this data set, e.g. type data contained.

z.di - this structure array contains *D*imension *I*nformation meta information, which gives the name of the dimension, the names for each of it's entries, the units of the dimension and any additional extra information about the dimension.  (see mkDimInfo) 

z.prep - this structure array contain *PREP*rocessing history for the data, which says what pre-processing functions have been applied to the data with what options.  Each element of the array describes things like the pre-processing *method* used, the options (opts) used when calling this method, and general *info* about the results of applying this method to the data.

z.Y - this structure (and it's associated dimension Info (z.Ydi)) contain information about the labelling for the trails/epochs in the main data store (z.X).


Overview of methods available
-----------------------------


Accessing data
--------------

jf_load:
Usage: z=jf_load(expt,subj,label,session,errorp)
Description:
Load a sliced/pre-processed data file for the given experiment, subject and session with the given user defined label.
Example:
>> z=jf_load('own_experiments/motor_imagery/movement_detection/trial_based/offline','linsey','trn','20110112');

jf_save:
Usage: jf_save(z,label)
Description:
Save the pre-processed data contained in z to it's correct location in the subjects experiment directory with the given label, such that it can be found again with jf_load
Example:
>> jf_save(z,'trn2');

Visualising data
----------------

jf_disp
-------
Usage: jf_disp(z)
Description:
Give a short textual summary of the data contained in z with it's pre-processing history.
Example:
>> jf_disp(z)
own_experiments/visual/rephrase 	 jelmer 	 20091113 	 (train_pp)
[58 ch_dss x 76 times x 907 epoch] of mV (single)
 1) 200911181743         raw2jf - Status= [201 205 208 209 213 214 215 216 221 225 232]->[201 205 208 209 213 214 215 216 221 225 232]
 2) 200911181743         jf_cat - 1 datasets along epoch
 3) 200911181743      jf_reject - reject 2 chs (EXG7,EXG8)  (unused)
 4) 201102072206     jf_detrend - over time 
 5) 201102072206 jf_spatdownsample - slap mapped 71 chs -> 71 ch_dss
Labels: [907 epoch x 1 subProb] of lab	(10 folds)


jf_plotERP
Usage: jf_plotERP(z)
Description:
Plot the average response of the data in z for each *class* in z.
Example:
>> jf_plotERP(z)
Notes:
Use jf_retain before the plot to sub-select the data to plot, e.g. to plot only the data between 1000ms and 2000ms after the epoch start use:
>> jf_plotERP(jf_retain(z,'dim','time','range','between','vals',[1000 2000]));


jf_plotAUC
Usage: jf_plotAUC(z)
Description:
Plot the per-binary sub-problem Area under the ROC curve for each feature of the input data.  This gives an indicatio of what features contain the most class relevant information in the data.
Example:
>> jf_plotAUC(z)
Notes:
as for jf_plotERP use jf_retain to sub-select the range displayed in the plot

jf_plot
Usage: jf_plot(z)
Description:
Plot *all* the data contained in z.
Notes:
For a full 'raw' dataset you probably don't want to do this as this will plot a line for every epoch.  To sub-select to, say, only the first 10 epochs, use jf_retain to pre-select these epochs as described above
Example:
>> jf_plot(z);


Selecting Data
--------------

jf_retain/jf_reject
Usage: z=jf_retain(z,'dim','time','range','between','vals',[1000 2000]);
Description:
Sub select a portion of the data contained in z.  jf_retain keeps only the matching portion, jf_reject removes only the matching portion
Example:
>> z=jf_retain(z,'dim','time','range','between','vals',[1000 2000]);


jf_rmOutliers
Usage: z=jf_rmOutliers(z,'dim','ch','thresh',3);
Description:
Identify elements along dimension 'dim' of the input which have excessively high variance (i.e. more than 3 std deviations above the mean variance) and remove them from the data set.
Example:
>> jf_rmOutliers(z,'dim','ch');


Processing Data
---------------

jf_reref/jf_baseline
Usage: z=jf_reref(z,'dim','ch');
Description:
Normalise the values of z along dimension 'dim' by subtracting the mean value.
Example:
>> z=jf_reref(z,'dim','ch');
Notes:
To use a weighted mean base on only a sub-set of the elements along to use to compute the average to subtract use:
>> z=jf_reref(z,'dim','ch','wght',{'P9' 'P10'});    % linked mastoids ref
>> z=jf_reref(z,'dim','time','wght',[-2000 -1000]); % reference to average value from 2-1s before marker


Classifying Data
----------------

jf_compKernel
Usage: z=jf_compKernel(z);
Description:
Compute the kernel of the input data.  This is a examples x examples matrix which is used in the kernel-based classifier training methods.


jf_cvtrain
Usage: z=jf_cvtrain(z);
Description:
Train a classifier on the different sub-problems contained in z, using cross-validation (default to 10-fold) to identify the optimal hyper-parameter.  The default is to use a l_2 regularised Kernel Logistic Regression classifier.
As it runs this function will print the results of the cross-validation with each new fold starting a new row, and within each fold the training/testing classification performance for each hyper-parameter printed on each line.  The final line then gives the average training/testing performance of all folds.

Example:
>> z=jf_cvtrain(jf_compKernel(z))
(out)	0.50/NA  	0.50/NA  	0.50/NA  	0.61/NA  	0.85/NA  	1.00/NA  	1.00/NA  	
(  1)	0.50/0.50	0.50/0.50	0.51/0.50	0.60/0.55	0.87/0.51	1.00/0.47	1.00/0.45	
(  2)	0.50/0.50	0.50/0.50	0.50/0.50	0.60/0.49	0.91/0.49	1.00/0.54	1.00/0.52	
(  3)	0.50/0.50	0.50/0.50	0.50/0.50	0.61/0.47	0.92/0.44	1.00/0.44	1.00/0.44	
.
.
-------------------------
(ave)	0.50/0.50	0.50/0.50	0.50/0.50	0.61/0.50	0.89/0.49	1.00/0.49	1.00/0.48	


jf_addClassInfo
Usage: z=jf_addClassInfo(z,'spType',{{'left' 'right'} {'left' 'foot'}});
Description:
Relabel the epochs in the input data to have a different set of sub-problems. Note: the names used to describe epochs are obtained from the markerdict which was setup during slicing.
Example: Group epochs to setup a pair of rest vs. move type sub-problems
>> z=jf_addClassInfo(z,'spType',{{{'rest3' 'rest4' 'rest5'} {'rh3' 'rh4' 'rh5'}}...
                         {{'rest3' 'rest4' 'rest5'} {'any3' 'any4' 'any5'}}},'summary','rest vs move'); 


Feature Extraction
------------------

jf_welchpsd
Usage: jf_welchpsd(z,'width_ms',200)
Description:
Compute the power spectral density for the input data, i.e. map from time domain to frequency domain using welch's method, with 200ms windows.  
Note: The spectral resolution is inversely related to the window width, thus 200ms = .2 s -> 1/.2 Hz = 5Hz frequency resolution.
Example:
>> z=jf_welchpsd(z)


jf_spectrogram
Usage: jf_spectrogram(z,'width_ms',250)
Description:
Compute a time-frequency decomposition of the input data using the short-time fourier transform, where a window of with with_ms is computed every with_ms/2 samples, i.e. this gives a frequency spectrum of the data with frequency resolution 1./with_ms Hz every width_ms/2 milliseconds.
Notes: This will inherently increase the dimension of your data by 1 as it is now: channels x time x frequencies x epochs
Example:
>> z=jf_spectrogram(z);
