/*
* Photoneo's API Example - FullAPIExample.cpp
* Defines the entry point for the console application.
* Demonstrates the extended functionality of PhoXi devices. This Example shows various ways how to connect to device.
* Contains the usage of retrieving all parameters of the device. Tests the software trigger and free run mode.
* Describes the exact steps needed to change device's settings, to handle and save received frame.
* Points out the correct way to disconnect the device from PhoXiControl.
*/

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <thread>
#include <atomic>
#if defined(_WIN32)
    #include <windows.h>
#elif defined (__linux__)
    #include <unistd.h>
#endif

#include "PhoXi.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector
#include <pybind11/operators.h>//operator



class PhoxiControl {

private:
	bool isconnected = false;
	pho::api::PPhoXi cam;
	int frameid = -1;
	bool isframeobtained = false;
	pho::api::PFrame frame;

private:
	bool isDeviceAvailable(pho::api::PhoXiFactory &factory, const std::string &serialNumber) {
		if (!factory.isPhoXiControlRunning()) {
			std::cout << "[!] PhoXi Control Software is not running." << std::endl;
			return false;
		}
		std::vector <pho::api::PhoXiDeviceInformation> deviceList = factory.GetDeviceList();
		if (deviceList.empty()) {
			std::cout << "[!] 0 devices found." << std::endl;
			return false;
		}
		bool isFound = false;
		for (std::size_t i = 0; i < deviceList.size(); ++i) {
			if (deviceList[i].HWIdentification == serialNumber) {
				isFound = true;
			}
		}
		return isFound;
	}

	pho::api::PPhoXi connectToDevice(pho::api::PhoXiFactory &factory, const std::string &serialNumber) {
		if (!factory.isPhoXiControlRunning()) {
			std::cout << "PhoXi Control Software is not running!" << std::endl;
			return 0;
		}
		pho::api::PPhoXi PhoXiDevice = factory.CreateAndConnect(serialNumber);
		return PhoXiDevice;
	}

	void configDevice(const pho::api::PPhoXi &PhoXiDevice, const std::string &resolution) {
		// Set trigger to "freerun" mode
		if (PhoXiDevice->TriggerMode != pho::api::PhoXiTriggerMode::Software) {
			if (PhoXiDevice->isAcquiring()) {
				if (!PhoXiDevice->StopAcquisition()) {
					throw std::runtime_error("Error in StopAcquistion");
				}
			}
			PhoXiDevice->TriggerMode = pho::api::PhoXiTriggerMode::Software;
			std::cout << "[*] Software mode was set." << std::endl;
			if (!PhoXiDevice->TriggerMode.isLastOperationSuccessful()) {
				throw std::runtime_error(PhoXiDevice->TriggerMode.GetLastErrorMessage().c_str());
			}
		}
		// Just send Texture and DepthMap
		pho::api::FrameOutputSettings currentOutputSettings = PhoXiDevice->OutputSettings;
		pho::api::FrameOutputSettings newOutputSettings = currentOutputSettings;
		newOutputSettings.SendPointCloud = true;
		newOutputSettings.SendNormalMap = true;
		newOutputSettings.SendDepthMap = true;
		newOutputSettings.SendConfidenceMap = true;
		newOutputSettings.SendTexture = true;
		PhoXiDevice->OutputSettings = newOutputSettings;
		// Configure the device resolution
		pho::api::PhoXiCapturingMode mode = PhoXiDevice->CapturingMode;
		if (resolution == "low") {
			mode.Resolution.Width = 1032;
			mode.Resolution.Height = 772;
		}
		else {
			mode.Resolution.Width = 2064;
			mode.Resolution.Height = 1544;
		}
		PhoXiDevice->CapturingMode = mode;
	}

	bool getFrame(const pho::api::PPhoXi &PhoXiDevice, pho::api::PFrame &FrameReturn) {
		// start device acquisition if necessary
		if (!PhoXiDevice->isAcquiring()) PhoXiDevice->StartAcquisition();
		// clear the current acquisition buffer
		PhoXiDevice->ClearBuffer();
		if (!PhoXiDevice->isAcquiring()) {
			std::cout << "[!] Your device could not start acquisition!" << std::endl;
			return false;
		}
		std::cout << "[*] Waiting for a frame." << std::endl;
		int frameId = PhoXiDevice->TriggerFrame();
		FrameReturn = PhoXiDevice->GetSpecificFrame(frameId);
		//FrameReturn = PhoXiDevice->GetFrame(pho::api::PhoXiTimeout::Infinity);
		if (!FrameReturn) {
			std::cout << "[!] Failed to retrieve the frame!" << std::endl;
			return false;
		}
		std::cout << "[*] A new frame is captured." << std::endl;
		//PhoXiDevice->StopAcquisition();
		return true;
	}

	bool checkFrame(const pho::api::PFrame& FrameIn) {
		if (FrameIn->Empty()) {
			std::cout << "Frame is empty.";
			return false;
		}
		if ((FrameIn->DepthMap.Empty()) || (FrameIn->Texture.Empty()) || 
			(FrameIn->PointCloud.Empty()) || (FrameIn->NormalMap.Empty())) 
			return false;
		return true;
	}

	/// <summary>
	/// connec to the sensor, and save the cam to a global variable
	/// </summary>
	/// <param name="serialno"> serial number </param>
	/// <param name="portno"> port number </param>
	/// <param name="resolution"> resolution "low" or "high" </param>
	/// <returns> phoxi cam object </returns>
	/// 
	/// <author> weiwei </author>
	/// <date> 20191128 </date>
	bool connect(std::string serialno, unsigned int portno, std::string resolution) {
		if (isconnected) {
			std::cout << "[!] The device has been connected. There is no need to connect again." << std::endl;
			return true;
		}
		if (resolution != "high" && resolution != "low") {
			std::cout << "[!] Resolution must be one of [\"low\", \"high\"]." << std::endl;
			return false;
		}
		// check if any connected device matches the requested serial number
		pho::api::PhoXiFactory factory;
		bool isFound = isDeviceAvailable(factory, serialno);
		if (!isFound) {
			std::cout << "[!] Requested device (serial number: " << serialno << ") not found!" << std::endl;
			return false;
		}
		// connect to the device
		cam = connectToDevice(factory, serialno);
		if (!cam->isConnected()) {
			std::cout << "[!] Could not connect to device." << std::endl;
			return false;
		}
		else {
			std::cout << "[*] Successfully connected to device." << std::endl;
			isconnected = true;
		}
		// configure the device
		configDevice(cam, resolution);
		return true;
	}

public:
	/// <summary>
	/// constructor
	/// </summary>
	/// <param name="serialno"></param>
	/// <param name="portno"></param>
	/// <param name="resolution"></param>
	/// <returns></returns>
	PhoxiControl(std::string serialno, unsigned int portno, std::string resolution) {
		connect(serialno, portno, resolution);
	}

	/// <summary>
	/// destructor
	/// </summary>
	/// <param name="serialno"></param>
	/// <param name="portno"></param>
	/// <param name="resolution"></param>
	/// <returns></returns>
	~PhoxiControl() {
		if (isconnected) {

		}
	}

	/// <summary>
	/// capture a frame, and save it to the global frame variable
	/// </summary>
	/// <returns></returns>
	/// 
	/// <author> weiwei </author>
	/// <date> 20191128 </date>
	bool captureframe() {
		bool success = getFrame(cam, frame);
		if (checkFrame(frame) && success) {
			isframeobtained = true;
			frameid += 1;
			std::cout << "A new frame is obtained. FrameID: " << frameid << std::endl;
			return true;
		}
		else {
			return false;
		}
	}

	int getframeid() {
		if (isframeobtained) {
			return frameid;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return -1;
		}
	}

	int getframewidth() {
		if (isframeobtained) {
			return frame->DepthMap.Size.Width;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	int getframeheight() {
		if (isframeobtained) {
			return frame->DepthMap.Size.Height;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	unsigned long getdepthmapdatasize() {
		if (isframeobtained) {
			return frame->DepthMap.GetDataSize() / sizeof(float);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return 0;
		}
	}

	std::vector<float> gettexture() {
		if (isframeobtained) {
			unsigned long datasize = getdepthmapdatasize();
			float* textureptr = (float*)frame->Texture.GetDataPtr();
			return std::vector<float>(textureptr, textureptr + datasize);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getdepthmap() {
		if (isframeobtained) {
			unsigned long datasize = getdepthmapdatasize();
			float* depthmapptr = (float*)frame->DepthMap.GetDataPtr();
			return std::vector<float>(depthmapptr, depthmapptr + datasize);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getpointcloud() {
		if (isframeobtained) {
			int height = frame->PointCloud.Size.Height;
			int width = frame->PointCloud.Size.Width;
			std::vector<float> result(height*width * 3);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					result[i*width*3+j*3+0] = frame->PointCloud[i][j].x;
					result[i*width*3+j*3+1] = frame->PointCloud[i][j].y;
					result[i*width*3+j*3+2] = frame->PointCloud[i][j].z;
				}
			}
			return result;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	std::vector<float> getnormals() {
		if (isframeobtained) {
			int height = frame->NormalMap.Size.Height;
			int width = frame->NormalMap.Size.Width;
			std::vector<float> result(height*width * 3);
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					result[i*width * 3 + j * 3 + 0] = frame->NormalMap[i][j].x;
					result[i*width * 3 + j * 3 + 1] = frame->NormalMap[i][j].y;
					result[i*width * 3 + j * 3 + 2] = frame->NormalMap[i][j].z;
				}
			}
			return result;
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
		}
	}

	// bool saveply(const std::string filepath, int xmin, int xmax, int ymin, int ymax, int zmin, int zmax) {
	bool saveply(const std::string filepath) {
		//revised --> delete x,y,z boundaries as parameters
		//by xinyi
		if (isframeobtained) {
			int height = frame->PointCloud.Size.Height;
			int width = frame->PointCloud.Size.Width;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float x = frame->PointCloud[i][j].x;
					float y = frame->PointCloud[i][j].y;
					float z = frame->PointCloud[i][j].z;
					// if ((x < xmin || x > xmax) || (y < ymin || y > ymax) || (z < zmin || z > zmax)) {
					if ((x < -99999 || x > 99999) || (y < -99999 || y > 99999) || (z < -99999 || z > 99999)) {
						frame->PointCloud[i][j].x = 0;
						frame->PointCloud[i][j].y = 0;
						frame->PointCloud[i][j].z = 0;
					}
				}
			}
			return frame->SaveAsPly(filepath, true, false, true, false, false, false, false, false);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return false;
		}
	}
	bool saveplyauto() {
		if (isframeobtained) {
			std::string filepath = "/tmp/out.ply";
			int xmin = -99999;
			int xmax = 99999;
			int ymin = -99999;
			int ymax = 99999;
			int zmin = -99999;
			int zmax = 99999;
			int height = frame->PointCloud.Size.Height;
			int width = frame->PointCloud.Size.Width;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					float x = frame->PointCloud[i][j].x;
					float y = frame->PointCloud[i][j].y;
					float z = frame->PointCloud[i][j].z;
					if ((x < xmin || x > xmax) || (y < ymin || y > ymax) || (z < zmin || z > zmax)) {
						frame->PointCloud[i][j].x = 0;
						frame->PointCloud[i][j].y = 0;
						frame->PointCloud[i][j].z = 0;
					}
				}
			}
			return frame->SaveAsPly(filepath, true, false, true, false, false, false, false, false);
		}
		else {
			std::cout << "Grap a frame first!" << std::endl;
			return false;
		}
	}
};

namespace py = pybind11;

PYBIND11_MODULE(phoxicontrol, m) {
	py::class_<PhoxiControl>(m, "PhoxiControl")
		.def(py::init<std::string, unsigned int, std::string>())
		.def("captureframe", &PhoxiControl::captureframe)
		.def("getframeid", &PhoxiControl::getframeid)
		.def("getframewidth", &PhoxiControl::getframewidth)
		.def("getframeheight", &PhoxiControl::getframeheight)
		.def("getdepthmapdatasize", &PhoxiControl::getdepthmapdatasize)
		.def("gettexture", &PhoxiControl::gettexture)
		.def("getdepthmap", &PhoxiControl::getdepthmap)
		.def("getpointcloud", &PhoxiControl::getpointcloud)
		.def("getnormals", &PhoxiControl::getnormals)
		.def("saveply", &PhoxiControl::saveply)
		.def("saveplyauto", &PhoxiControl::saveplyauto);
//		.def("findmodel", &PhoxiControl::findmodel);
}