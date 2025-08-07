import glob
import os
import shutil
import math

import numpy

import azcam
import azcam.utils
import azcam.fits
import azcam_console.utils
import azcam.image
from azcam_console.testers.basetester import Tester
from . import robust_stats


class Prnu(Tester):
    """
    Photo-Response Non-Uniformity (PRNU) acquisition and analysis.
    """

    def __init__(self):
        super().__init__("prnu")

        self.exposure_type = "flat"
        self.root_name = "prnu."  # allow for analyzing QE data
        self.allowable_deviation_from_mean = -1  # allowable deviation from mean signal
        self.exposure_levels = {}  # dictionary of {wavelength:exposure times}
        self.mean_count_goal = 0  # use detcal data if > 0
        self.grades = {}  # Pass/Fail grades at each wavelength {wave:grade}

        self.fit_order = 3
        """fit order for overscan correction"""

        self.bias_image_in_sequence = 1  # flag true if first image is a bias image

        self.overscan_correct = 0  # flag to overscan correct images
        self.zero_correct = 0  # flag to correct with bias residuals

        self.roi_prnu = []  # empty to use entire sensor

        # outputs
        self.prnu_file = ""
        self.prnus = {}  # PRNU at each image in sequence {wavelength:PRNU}

        self.data_file = "prnu.txt"
        self.report_file = "prnu"

    def acquire(self):
        """
        Acquire a bias image and one or more flats for PRNU analysis.
        ExposureTimes and Wavelengths are read from ExposuresFile text file.
        """

        azcam.log("Acquiring PRNU sequence")

        # save pars to be changed
        impars = {}
        azcam.utils.save_imagepars(impars)

        # create new subfolder
        currentfolder, subfolder = azcam_console.utils.make_file_folder("prnu")
        azcam.db.parameters.set_par("imagefolder", subfolder)

        # clear device
        azcam.db.parameters.set_par("imagetest", 1)
        imname = "test.fits"
        azcam.db.tools["exposure"].test(0)
        bin1 = int(azcam.fits.get_keyword(imname, "CCDBIN1"))
        bin2 = int(azcam.fits.get_keyword(imname, "CCDBIN2"))
        binning = bin1 * bin2

        azcam.db.parameters.set_par("imageroot", "prnu.")  # for automatic data analysis
        azcam.db.parameters.set_par(
            "imageincludesequencenumber", 1
        )  # use sequence numbers
        azcam.db.parameters.set_par("imageautoname", 0)  # manually set name
        azcam.db.parameters.set_par(
            "imageautoincrementsequencenumber", 1
        )  # inc sequence numbers
        azcam.db.parameters.set_par("imagetest", 0)  # turn off TestImage

        # bias image
        azcam.db.parameters.set_par("imagetype", "zero")
        filename = os.path.basename(azcam.db.tools["exposure"].get_filename())
        azcam.log("Taking PRNU bias: %s" % filename)
        azcam.db.tools["exposure"].expose(0, "zero", "PRNU bias")

        # exposure times
        waves = list(self.exposure_levels.keys())
        waves.sort()
        if self.mean_count_goal > 0:
            if azcam.db.tools["detcal"].is_valid:
                for wave in waves:
                    meancounts = azcam.db.tools["detcal"].mean_counts[wave]
                    self.exposure_levels[wave] = (
                        (self.mean_count_goal / meancounts)
                        / binning
                        * (
                            azcam.db.tools["gain"].system_gain[0]
                            / azcam.db.tools["detcal"].system_gain[0]
                        )
                    )

            else:
                azcam.log("invalid detcal data, using fixed exposure times")

        for wave in waves:
            wavelength = float(wave)
            exposuretime = self.exposure_levels[wave]

            if wavelength > 0:
                azcam.log(f"Moving to wavelength: {int(wavelength)}")
                azcam.db.tools["instrument"].set_wavelength(wavelength)
                wave = azcam.db.tools["instrument"].get_wavelength()
                wave = int(wave)
                azcam.log(f"Current wavelength: {wave}")
            filename = os.path.basename(azcam.db.tools["exposure"].get_filename())
            azcam.log(
                f"Taking PRNU image for {exposuretime:.3f} seconds at {wavelength:.1f} nm"
            )
            azcam.db.tools["exposure"].expose(
                exposuretime, self.exposure_type, f"PRNU image {wavelength:.1f} nm"
            )

        # finish
        azcam.utils.restore_imagepars(impars)
        azcam.utils.curdir(currentfolder)
        azcam.log("PRNU sequence finished")

        return

    def analyze(self, copy_files=0):
        """
        Analyze an existing PRNU image sequence.
        """

        azcam.log("Analyzing PRNU sequence")

        rootname = self.root_name
        self.grade = "UNDEFINED"
        subfolder = "analysis"
        self.images = []

        # analysis subfolder
        startingfolder, subfolder = azcam_console.utils.make_file_folder(subfolder)
        azcam.log("Making copy of image files for analysis")
        for filename in glob.glob(os.path.join(startingfolder, "*.fits")):
            shutil.copy(filename, subfolder)
        azcam.utils.curdir(subfolder)
        currentfolder = azcam.utils.curdir()
        _, StartingSequence = azcam_console.utils.find_file_in_sequence(rootname)
        SequenceNumber = StartingSequence

        # bias image (nominally first in sequence)
        zerofilename = rootname + "%04d" % StartingSequence
        zerofilename = os.path.join(currentfolder, zerofilename) + ".fits"
        zerofilename = azcam.utils.make_image_filename(zerofilename)

        numext, _, _ = azcam.fits.get_extensions(zerofilename)
        if numext == 0:
            numext = 1

        if self.bias_image_in_sequence:
            SequenceNumber += 1

        nextfile = os.path.normpath(
            os.path.join(currentfolder, rootname + "%04d" % (SequenceNumber)) + ".fits"
        )

        # get gain values
        if azcam.db.tools["gain"].is_valid:
            self.system_gain = azcam.db.tools["gain"].system_gain
        else:
            azcam.log("WARNING: no gain values found for scaling")
            self.system_gain = numext * [1.0]

        # loop over files
        self.grades = {}
        while os.path.exists(nextfile):
            wavelength = azcam.fits.get_keyword(nextfile, "WAVLNGTH")
            wavelength = int(float(wavelength) + 0.5)
            azcam.log("Processing image %s" % os.path.basename(nextfile))

            # colbias
            if self.overscan_correct:
                azcam.fits.colbias(nextfile, fit_order=self.fit_order)

            # "debias" correct with residuals after colbias
            # Global offsets are taken care of later using gain.zero_mean
            if self.zero_correct:
                debiased = azcam.db.tools["bias"].debiased_filename
                biassub = "biassub.fits"
                azcam.fits.sub(nextfile, debiased, biassub)
                os.remove(nextfile)
                os.rename(biassub, nextfile)

            # scale to electrons by system gain
            prnuimage = azcam.image.Image(nextfile)

            zmean = None if self.overscan_correct else azcam.db.tools["gain"].zero_mean
            prnuimage.set_scaling(self.system_gain, zmean)
            prnuimage.assemble(1)
            prnuimage.save_data_format = -32
            prnuimage.write_file("prnu_%d.fits" % wavelength, 6)

            # create masked array
            self.masked_image = numpy.ma.array(prnuimage.buffer, mask=False)
            defects = azcam.db.tools["defects"]
            defects.mask_defects(self.masked_image)

            # apply defects mask
            self.masked_image = numpy.ma.array(prnuimage.buffer, mask=False)
            defects = azcam.db.tools["defects"]
            defects.mask_edges(self.masked_image)

            # optionally use ROI
            if len(self.roi_prnu) == 1:
                x1, x2, y1, y2 = self.roi_prnu
                maskedimage = self.masked_image[y1:y2, x1:x2]
                stdev = numpy.ma.std(maskedimage)
                mean = numpy.ma.mean(maskedimage)

                # account for signal shot noise
                prnu = math.sqrt(stdev**2 - mean) / mean
                if numpy.isnan(prnu):
                    prnu = 0.0
            else:
                # stdev = numpy.ma.std(self.masked_image)
                # mean = numpy.ma.mean(self.masked_image)
                # azcam.log("PRNU: mean = %f, stdev = %f" % (mean, stdev))

                # vals = self.masked_image.compressed()
                # stdev = robust_stats.medabsdev(vals)
                # mean = numpy.median(vals)
                # azcam.log("Robust PRNU: mean = %f, stdev = %f" % (mean, stdev))

                # PRNU for each channel
                ys, xs = self.masked_image.shape
                if numext == 1:
                    x1_arr = [0]
                    x2_arr = [xs]
                    y1_arr = [0]
                    y2_arr = [ys]
                elif numext == 2:
                    x1_arr = [0, xs//2]
                    x2_arr = [xs//2, xs]
                    y1_arr = [0, 0]
                    y2_arr = [ys, ys]
                elif numext == 4:
                    x1_arr = [0, 0, xs//2, xs//2]
                    x2_arr = [xs//2, xs//2, xs, xs]
                    y1_arr = [0, ys//2, 0, ys//2]
                    y2_arr = [ys//2, ys, ys//2, ys]
                else:
                    # Assume channels are evenly split on top and bottom
                    x1_arr = numpy.linspace(0, xs, numext//2 + 1)[:-1].astype(int)
                    x1_arr = 2 * x1_arr.to_list()  # double for top and bottom
                    x2_arr = numpy.linspace(0, xs, numext//2 + 1)[1:].astype(int)
                    x2_arr = 2 * x2_arr.to_list()  # double for top and bottom
                    y1_arr = (numext//2) * [0]     + (numext//2) * [ys//2]
                    y2_arr = (numext//2) * [ys//2] + (numext//2) * [ys]


                # x1_arr = [0, 0, 2048, 2048]
                # x2_arr = [2048, 2048, 4096, 4096]
                # y1_arr = [0, 2048, 0, 2048]
                # y2_arr = [2048, 4096, 2048, 4096]
                prnu_arr = []
                for i in range(numext):
                    x1 = x1_arr[i]
                    x2 = x2_arr[i]
                    y1 = y1_arr[i]
                    y2 = y2_arr[i]
                    maskedimage = self.masked_image[y1:y2, x1:x2]
                    # stdev = numpy.ma.std(maskedimage)
                    # mean = numpy.ma.mean(maskedimage)
                    # prnu = math.sqrt(stdev**2 - mean) / mean
                    # azcam.log(
                    #     "  PRNU: mean = %f, stdev = %f, prnu = %f"
                    #     % (mean, stdev, prnu)
                    # )
                    vals = maskedimage.compressed()
                    stdev = robust_stats.medabsdev(vals)
                    mean = numpy.median(vals)
                    prnu = math.sqrt(stdev**2 - mean) / mean
                    prnu_arr.append(prnu)
                    # azcam.log(
                    #     "  Robust PRNU: mean = %f, stdev = %f, prnu = %f"
                    #     % (mean, stdev, prnu)
                    # )

                prnu = prnu_arr[0] if numext==1 else numpy.nanmean(prnu_arr)
                if numpy.isnan(prnu):
                    prnu = 0.0


            self.prnus[wavelength] = float(prnu)
            if self.allowable_deviation_from_mean != -1:
                if prnu <= self.allowable_deviation_from_mean:
                    GRADE = "PASS"
                else:
                    GRADE = "FAIL"
            else:
                GRADE = "UNDEFINED"

            self.grades[wavelength] = GRADE

            s = "PRNU at %7.1f nm is %5.1f%%, Grade = %s" % (
                wavelength,
                prnu * 100,
                GRADE,
            )
            azcam.log(s)

            SequenceNumber = SequenceNumber + 1
            nextfile = (
                os.path.join(currentfolder, rootname + "%04d" % SequenceNumber)
                + ".fits"
            )

        if "FAIL" in list(self.grades.values()):
            self.grade = "FAIL"
        else:
            self.grade = "PASS"
        s = "Grade = %s" % self.grade

        if not self.grade_sensor:
            self.grade = "UNDEFINED"

        azcam.log(s)

        # define dataset
        self.dataset = {
            "data_file": self.data_file,
            "grade": self.grade,
            "allowable_deviation_from_mean": self.allowable_deviation_from_mean,
            "Prnus": self.prnus,
            "grades": self.grades,
        }

        # write data file
        self.write_datafile()
        if self.create_reports:
            self.report()

        # copy data files
        files = ["prnu.txt", "prnu.md", "prnu.pdf"]
        for f in files:
            shutil.copy(f, startingfolder)

        # write data file
        azcam.utils.curdir(startingfolder)
        self.write_datafile()
        if self.create_reports:
            self.report()

        # finish
        self.is_valid = True
        return

    def report(self):
        """
        Write dark report file.
        """

        lines = ["# PRNU Analysis"]

        if self.allowable_deviation_from_mean != -1:
            if self.grade != "UNDEFINED":
                s = f"PRNU spec= {(self.allowable_deviation_from_mean * 100.0):.1f}%  "
                lines.append(s)
                s = f"PRNU grade = {self.grade}  "
                lines.append(s)

        lines.append("")
        s = "|**Wavelength**|**PRNU [%]**|"
        lines.append(s)
        s = "|:---|:---:|"
        lines.append(s)

        waves = list(self.prnus.keys())
        waves.sort()
        for wave in waves:
            s = f"|{wave:04d}|{(100.0 * self.prnus[wave]):7.01f}|"
            lines.append(s)

        # Make report files
        self.write_report(self.report_file, lines)

        return
