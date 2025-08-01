"""
Quantum Efficiency (QE) tester for azcam.
This version uses Power Meter (not diodes) for calibration.
"""

import glob
import os
import shutil

import numpy

import azcam
import azcam.utils
import azcam.fits
import azcam.image
import azcam_console.plot
from azcam_console.testers.basetester import Tester
from . import robust_stats


class QE(Tester):
    """
    Quantum Efficiency (QE) acquisition and analysis.
    """

    def __init__(self):
        super().__init__("qe")

        self.flux_cal_folder = ""
        self.flux_cal_file = "flux_cal.txt"

        self.window_trans = {}  # dictionary of {wave:trans}
        self.throughputs = []  # effective throughput, usually only window
        self.qe_specs = {}  # QE specifications for PASS/FAIL {wavelength:spec}
        self.grades = {}  # QE grade {wavelength:grade}

        self.use_exposure_levels = (
            0  # 1 to use exposure levels, else use exposure times
        )
        self.exposure_levels = {}  # Exposure levels {wave:level} [DN/pix]
        self.exposure_times = (
            {}
        )  # Exposure times {wave:seconds]  (when no exposure_levels)
        self.exptime_offset = 0.0

        self.means = []  # mean counts
        self.qe = {}  # QE values
        self.wavelengths = []  # Wavelengths for QE measurements
        self.fluxes = []  # photons/sec/mm^2 at reference
        self.flush_before_exposure = 0  # number of extra flush before each exposure

        self.include_dark_images = 0  # include dark images in acquire & analysis

        self.overscan_correct = 1
        self.zero_correct = 0

        self.system_gain = []

        self.fit_order = 3
        """fit order for overscan correction"""

        self.diode_area = 613.0  # diode area in mm^2
        self.diode_wavelength = []
        self.sphere_current = []
        self.diode_power = []
        self.diode_current = []
        self.diode_qe = {}

        self.cal_scale = 1.0  # geometric scaling from reference to sensor
        self.pixel_area = 0.015 * 0.015  # unbinned pixel area in mm^2
        self.mean_temp = -999  # Mean temperature of data
        self.binning = 1
        self.global_scale = 1.0  # scale factor

        self.plot_limits = []  # min and max of plot
        self.plot_title = ""  # title for QE plot
        self.use_errorbars = 1

        self.data_file = "qe.txt"
        self.report_file = "qe"

        self.qeroi = []  # special ROI for qe [row1,row2,col1,col2]

    def acquire(self):
        """
        Acquire a series of flats for QE measurement.
        Assumes timing code and binning is set as desired.
        """

        azcam.log("Acquiring QE sequence")

        exposure, instrument, detcal = azcam_console.utils.get_tools(
            ["exposure", "instrument", "detcal"]
        )

        # save pars to be changed
        impars = {}
        azcam.utils.save_imagepars(impars)

        # create new subfolder
        currentfolder, subfolder = azcam_console.utils.make_file_folder("qe")
        azcam.db.parameters.set_par("imagefolder", subfolder)

        # clear device
        azcam.db.tools["exposure"].test(0)
        imname = "test.fits"
        bin1 = int(azcam.fits.get_keyword(imname, "CCDBIN1"))
        bin2 = int(azcam.fits.get_keyword(imname, "CCDBIN2"))
        binning = bin1 * bin2

        azcam.db.parameters.set_par("imageroot", "qe.")  # for automatic data analysis
        azcam.db.parameters.set_par(
            "imageincludesequencenumber", 1
        )  # use sequence numbers
        azcam.db.parameters.set_par(
            "imagesequencenumber", 1
        )  # start at sequence number 1
        azcam.db.parameters.set_par("imageautoname", 0)  # manually set name
        azcam.db.parameters.set_par(
            "imageautoincrementsequencenumber", 1
        )  # inc sequence numbers
        azcam.db.parameters.set_par("imagetest", 0)  # turn off TestImage

        exposure.roi_reset()  # use entire device

        # get exposure times
        if self.use_exposure_levels:
            azcam.log("Using exposure_levels")

            self.exposure_times = {}  # reset
            for w in self.exposure_levels:
                meancounts = azcam.db.tools["detcal"].mean_counts[w]
                et = self.exposure_levels[w] / meancounts / binning
                et = et * (
                    azcam.db.tools["gain"].system_gain[0]
                    / azcam.db.tools["detcal"].system_gain[0]
                )

                self.exposure_times[w] = et

        else:
            azcam.log("Using exposure_times")

        # take bias image
        azcam.db.parameters.set_par("imageroot", "qe.")
        azcam.log("Taking bias image %s..." % os.path.basename(exposure.get_filename()))

        exposure.expose(0, "zero", "QE bias")

        for wave in self.exposure_times:
            wave = int(0.5 + float(wave))  # make sure wave is an integer

            etime = self.exposure_times[wave]
            title = f"{wave} nm QE flat for {etime} secs"
            instrument.set_wavelength(wave)

            azcam.log(
                f"Taking {wave} nm QE image for {etime:0.03f}seconds: {os.path.basename(exposure.get_filename())}"
            )

            # make exposure
            for _ in range(self.flush_before_exposure):
                exposure.test(0)

            if self.include_dark_images:
                darktitle = f"dark image for {etime} secs"
                exposure.expose(etime, "dark", f"{darktitle}")

            exposure.expose(etime, "flat", f"{title}")

        # copy flux cal file to local folder
        try:
            f1 = os.path.join(self.flux_cal_folder, self.flux_cal_file)
            f2 = os.path.join(subfolder, "flux_cal.txt")
            shutil.copyfile(f1, f2)
        except FileNotFoundError:
            pass

        # finish
        azcam.utils.restore_imagepars(impars)
        azcam.utils.curdir(currentfolder)

        return

    def analyze(self):
        """
        Analyze an exisiting QE series of flats.
        Includes Newport power meter calibration.
        """

        h = 6.62607015e-34  # J⋅s
        c = 2.99792458e8  # m/s

        azcam.log("Analyzing QE sequence")

        rootname = "qe."
        subfolder = "analysis"

        # read DiodeCalibration file
        self.diode_wavelength = []
        self.sphere_current = []
        self.diode_power = []

        with open(self.flux_cal_file, "r") as df:
            for line in df.readlines():
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                tokens = line.split("\t")
                self.diode_wavelength.append(int(float(tokens[0]) + 0.5))
                self.diode_power.append(float(tokens[1]))

        # copy all image files to analysis folder
        startingfolder, subfolder = azcam_console.utils.make_file_folder(subfolder)
        azcam.log("Making copy of image files for analysis")
        for filename in glob.glob(os.path.join(startingfolder, "*.fits")):
            shutil.copy(filename, subfolder)

        azcam.utils.curdir(subfolder)

        _, StartingSequence = azcam_console.utils.find_file_in_sequence(rootname)
        SequenceNumber = StartingSequence

        # get gain
        if azcam.db.tools["gain"].is_valid:
            self.system_gain = azcam.db.tools["gain"].system_gain
        else:
            azcam.log("WARNING: no gain values found for scaling")

        # bias level
        zerofilename = rootname + "%04d" % SequenceNumber
        zerofilename = os.path.join(subfolder, zerofilename) + ".fits"
        zmeans = azcam.fits.mean(zerofilename)

        try:
            bin1 = int(azcam.fits.get_keyword(zerofilename, "CCDBIN1"))
            bin2 = int(azcam.fits.get_keyword(zerofilename, "CCDBIN2"))
            binning = bin1 * bin2
            azcam.log(f"Binning is {binning} pixels ({bin1}x{bin2})")
        except Exception as e:
            azcam.log(e)
            binning = 1  # assume no keyword means no binning

        nextfile = zerofilename  # just to start loop
        SequenceNumber = SequenceNumber + 1

        # loop through image files
        self.exposures = []  # exposure times as read from data
        self.throughputs = []
        self.wavelengths = []  # wavelengths as read from data
        self.qe = {}
        self.fluxes = []
        self.means = []
        while os.path.exists(nextfile):
            if self.include_dark_images:
                darkfilename = rootname + "%04d" % (SequenceNumber)
                darkfilename = os.path.join(subfolder, darkfilename) + ".fits"

                qefilename = rootname + "%04d" % (SequenceNumber + 1)
                qefilename = os.path.join(subfolder, qefilename) + ".fits"
            else:
                qefilename = rootname + "%04d" % SequenceNumber
                qefilename = os.path.join(subfolder, qefilename) + ".fits"

            wave = 0
            exptime = 0
            meantemp = -999

            try:
                exptime = float(azcam.fits.get_keyword(qefilename, "EXPTIME"))
                wave = float(azcam.fits.get_keyword(qefilename, "WAVLNGTH"))
                wave = int(float(wave) + 0.5)
            except Exception:
                # try wavelength in OBJECT keyword for manual testing
                s = azcam.fits.get_keyword(qefilename, "OBJECT")
                wave = s.split(" ")[0]
                wave = int(float(wave) + 0.5)

            self.exposures.append(exptime)

            try:
                meantemp = float(azcam.fits.get_keyword(qefilename, "CAMTEMP"))
            except Exception:
                meantemp = -999.0

            for idiode, w in enumerate(self.diode_wavelength):
                if int(w) >= int(wave):
                    break

            windowwaves = sorted(self.window_trans.keys())
            for w in windowwaves:
                if int(float(w) + 0.5) >= int(wave):
                    windowstrans = float(self.window_trans[w])
                    break

            self.throughputs.append(windowstrans)

            # bias or dark correct
            if self.include_dark_images:
                azcam.fits.sub(qefilename, darkfilename, qefilename)
            elif self.overscan_correct:
                azcam.fits.colbias(qefilename, fit_order=self.fit_order)

            # scale to electrons by system gain
            qeimage = azcam.image.Image(qefilename)

            if self.overscan_correct or self.include_dark_images:
                qeimage.set_scaling(self.system_gain, None)
            else:
                qeimage.set_scaling(self.system_gain, zmeans)
            qeimage.assemble(1)

            # write scaled images as fits files
            qeimage.save_data_format = -32
            qeimage.write_file(f"qeimage_{wave}_{exptime:03f}.fits", 6)

            # create masked array
            self.masked_image = numpy.ma.array(qeimage.buffer, mask=False)
            defects = azcam.db.tools["defects"]
            defects.mask_defects(self.masked_image)

            if len(self.qeroi) == 0:
                vals = self.masked_image.compressed()
                qemean = robust_stats.mean(vals, Cut=10)
                # qemean = self.masked_image.mean()
            else:
                maskedimage = self.masked_image[
                    self.qeroi[2] : self.qeroi[3], self.qeroi[0] : self.qeroi[1]
                ]
                qemean = numpy.ma.mean(maskedimage)

            # calculate diode power [W/cm]
            dpower = self.diode_power[idiode]
            # compute flux in photons/mm^2/sec from W/cm^2
            ephoton = h * c / (wave * 1.0e-9)
            dpower = dpower / ephoton / 100
            self.fluxes.append(dpower)

            detectorcurrent = (
                qemean / (exptime + self.exptime_offset)
            ) / binning  # e/pixel/sec
            detectorcurrent = detectorcurrent / self.pixel_area  # e/mm^2/sec

            qe = (detectorcurrent / dpower) * self.cal_scale / windowstrans

            # QY correction
            qy = min(1.0, wave / 340.0)
            qe = qe * qy

            # global scale
            qe = qe * self.global_scale

            azcam.log(f"QE [{wave} nm] = {qe:.3f}")

            self.means.append(qemean)

            self.wavelengths.append(wave)
            self.qe[wave] = float(qe)

            SequenceNumber = SequenceNumber + 1
            if self.include_dark_images:
                SequenceNumber = SequenceNumber + 1
            nextfile = (
                os.path.join(subfolder, rootname + "%04d" % SequenceNumber) + ".fits"
            )

        # analyze grades for each wavelength
        for wave in self.wavelengths:
            if self.qe_specs != {}:
                try:
                    if self.qe[wave] < self.qe_specs[wave]:
                        self.grade = "FAIL"
                        self.grades[wave] = "FAIL"
                    else:
                        self.grades[wave] = "PASS"
                except KeyError:
                    self.grades[wave] = "UNDEFINED"
            else:
                self.grades[wave] = "UNDEFINED"

        if self.grade_sensor:
            if "FAIL" in self.grades.values():
                self.grade = "FAIL"
            else:
                self.grade = "PASS"
            azcam.log("Grade = %s" % self.grade)
        else:
            self.grade = "UNDEFINED"

        self.mean_temp = meantemp  # not mean yet

        # plot results
        if self.create_plots:
            self.plot()

        # copy processed files to starting folder
        if startingfolder != subfolder:
            try:
                shutil.copy("qe.png", startingfolder)
            except Exception:
                pass

        # define dataset
        # "Flux @ sensor is Flux*Throughput/CalScal"
        # "Flux is [photons/sec/mm^2@diode]"

        self.dataset = {
            "data_file": self.data_file,
            "grade": self.grade,
            "cal_scale": self.cal_scale,
            "mean_temp": str(self.mean_temp),
            "wavelengths": self.wavelengths,
            "qe": self.qe,
            "means": numpy.array(self.means).tolist(),
            "exposures": self.exposures,
            "fluxes": self.fluxes,
            "throughputs": self.throughputs,
            "system_gain": self.system_gain,
        }

        # write files
        azcam.utils.curdir(startingfolder)
        self.write_datafile()
        if self.create_reports:
            self.report()

        self.is_valid = True

        return

    def plot(self):
        """
        Plot QE data.
        """

        # setup figure
        bigfont = 18
        pbottom = 0.13
        ptop = 0.88
        pleft = 0.15
        pright = 0.95
        wspace = 0.2
        hspace = 0.2

        # make figure
        fig = azcam_console.plot.plt.figure()
        fignum = fig.number
        azcam_console.plot.move_window(fignum)
        if self.plot_title == "":
            fig.text(
                0.55,
                0.91,
                "Quantum Efficiency",
                horizontalalignment="center",
                fontsize=bigfont,
            )
        else:
            fig.text(
                0.55,
                0.91,
                f"{self.plot_title}",
                horizontalalignment="center",
                fontsize=bigfont,
            )
        fig.subplots_adjust(
            left=pleft,
            bottom=pbottom,
            right=pright,
            top=ptop,
            wspace=wspace,
            hspace=hspace,
        )
        ax = azcam_console.plot.plt.gca()
        ax.grid(1)
        azcam_console.plot.plt.xlabel("Wavelength [nm]", fontsize=bigfont)
        azcam_console.plot.plt.ylabel("Measured QE", fontsize=bigfont)

        ax.yaxis.set_major_locator(azcam_console.plot.plt.MaxNLocator(11))
        x = 2 * max(self.wavelengths) - min(self.wavelengths) + 1
        x = int(x / 100.0)
        ax.xaxis.set_major_locator(azcam_console.plot.plt.MaxNLocator(x))

        if self.mean_temp != -999:
            labels = [f"Mean Temp = {self.mean_temp:.0f} C"]
            ax.annotate(
                labels[0],
                xy=(0.6, 0.1),
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="top",
            )

        # plot data
        waves = self.wavelengths
        qevals = []
        for w in waves:
            qevals.append(self.qe[w])
        if self.use_errorbars:
            azcam_console.plot.plt.errorbar(
                waves, [x * 100.0 for x in qevals], yerr=3.0, marker="o", ls=""
            )
        else:
            azcam_console.plot.plt.plot(waves, [x * 100.0 for x in qevals], "bo-")

        if len(self.plot_limits) == 2:
            azcam_console.plot.plt.xlim(self.plot_limits[0][0], self.plot_limits[0][1])
            azcam_console.plot.plt.ylim(self.plot_limits[1][0], self.plot_limits[1][1])
        elif len(self.plot_limits) == 1:
            azcam_console.plot.plt.xlim(self.plot_limits[0][0], self.plot_limits[0][1])
        else:
            pass

        # plot specs
        if len(self.qe_specs) > 0:
            for wave in self.qe_specs:
                if self.qe_specs[wave] > 0:
                    x = wave
                    y = self.qe_specs[wave] * 100.0
                    azcam_console.plot.plt.plot(
                        x, y, ls="", marker="_", markersize=5, color="red"
                    )

        # save figure
        azcam_console.plot.plt.show()
        azcam_console.plot.save_figure(fignum, "qe.png")

        return

    def report(self):
        """
        Make report files.
        """

        QEPLOT = "qe.png"

        lines = ["# Quantum Efficiency Analysis"]

        if self.grade != "UNDEFINED":
            s = "QE grade = %s  " % self.grade
            lines.append(s)

        lines.append(f"![QE Plot]({os.path.abspath(QEPLOT)})  ")
        lines.append("*Quantum Efficiency Plot.*  ")

        lines.append("")
        if self.grade == "UNDEFINED":
            s = "|**Wavelength**|**QE**|"
            lines.append(s)
            s = "|:---|:---:|"
            lines.append(s)
        else:
            s = "|**Wavelength**|**QE**|**QE Spec.**|**Grade**|"
            lines.append(s)
            s = "|:---|:---:|:---:|:---:|"
            lines.append(s)

        for wave in self.wavelengths:
            if self.qe_specs != {}:
                try:
                    spec = self.qe_specs[wave]
                except KeyError:
                    spec = 0
            else:
                spec = 0
            grade = "" if self.grades[wave] == "UNDEFINED" else self.grades[wave]
            if spec == 0:
                s = f"{wave}|{self.qe[wave]:5.03f}|"
            else:
                s = f"{wave}|{self.qe[wave]:5.03f}|{spec:5.03f}|{grade}|"
            lines.append(s)

        # Make report files
        self.write_report(self.report_file, lines)

        return
