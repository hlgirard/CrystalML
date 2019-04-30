# CrystalML

![Travis CI status](https://travis-ci.com/hlgirard/CrystalML.svg?branch=master)

Integrated tool to measure the nucleation rate of protein crystals from the crystallization kinetics of an array of independent identical droplets.

From a directory containing a time-series of images of mutliple droplets, the tool segments individual droplet and uses a pre-trained CNN model to determine the presence or absence of crystals in each drop.
The nucleation rate is evaluated from the rate of decay of the proportion of drops that _do not_ exhibit visible crystals.

![Schematic](docs/CrystalML_demo.jpg)

## Repository strutcture

- `bin`: useful scripts
- `models`: pre-trained machine learning models for crystal presence discrimination
- `notebooks`: jupyer notebooks evaluating different image segmentation strategies
- `src`: source code for the project
    - `crystal_processing`: processing pipeline from directory to nucleation rate
    - `data`: data processing methods, including cropping, segmentation, extraction
    - `models`: model definition and training scripts for the droplet binary labelling task
    - `visualization`: visualization and plotting methods
- `tests`: unittesting

## License

This project is licensed under the GPLv3 License - see the [LICENSE.md](LICENSE.md) file for details.

## Credit

Initial models were built starting from the example at:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Live data visualization class TrainingPlot originally from:
https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
