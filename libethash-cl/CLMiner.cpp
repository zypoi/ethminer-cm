/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#include "CLMiner.h"
#include <libethash/internal.h>
#include "CLMiner_kernel.h"
#include <boost/algorithm/string.hpp>

#include "CLASM_baffin.h"
#include "CLASM_ellesmere1.h"
#include "CLASM_ellesmere2.h"
#include "CLASM_fiji.h"
#include "CLASM_hawaii.h"
#include "CLASM_pitcarin.h"
#include "CLASM_tahiti.h"
#include "CLASM_tonga1.h"
#include "CLASM_tonga2.h"
#include "CLASM_vega.h"

//#include "CLASM_generic.h"

using namespace dev;
using namespace eth;

namespace dev
{
namespace eth
{

unsigned CLMiner::s_workgroupSize = CLMiner::c_defaultLocalWorkSize;
unsigned CLMiner::s_initialGlobalWorkSize = CLMiner::c_defaultGlobalWorkSizeMultiplier * CLMiner::c_defaultLocalWorkSize;
unsigned CLMiner::s_asmVersion = 0;
unsigned CLMiner::s_ethIntensity = 8; 

constexpr size_t c_maxSearchResults = 1;

struct CLChannel: public LogChannel
{
	static const char* name() { return EthOrange " cl"; }
	static const int verbosity = 2;
	static const bool debug = false;
};
#define cllog clog(CLChannel)
#define ETHCL_LOG(_contents) cllog << _contents

namespace
{

void addDefinition(string& _source, char const* _id, unsigned _value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", _id, _value);
	_source.insert(_source.begin(), buf, buf + strlen(buf));
}

std::vector<cl::Platform> getPlatforms()
{
	vector<cl::Platform> platforms;
	try
	{
		cl::Platform::get(&platforms);
	}
	catch(cl::Error const& err)
	{
#if defined(CL_PLATFORM_NOT_FOUND_KHR)
		if (err.err() == CL_PLATFORM_NOT_FOUND_KHR)
			cwarn << "No OpenCL platforms found";
		else
#endif
			throw err;
	}
	return platforms;
}

std::vector<cl::Device> getDevices(std::vector<cl::Platform> const& _platforms, unsigned _platformId)
{
	vector<cl::Device> devices;
	size_t platform_num = min<size_t>(_platformId, _platforms.size() - 1);
	try
	{
		_platforms[platform_num].getDevices(
			CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
			&devices
		);
	}
	catch (cl::Error const& err)
	{
		// if simply no devices found return empty vector
		if (err.err() != CL_DEVICE_NOT_FOUND)
			throw err;
	}
	return devices;
}

}

}
}

unsigned CLMiner::s_platformId = 0;
unsigned CLMiner::s_numInstances = 0;
int CLMiner::s_devices[16] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

CLMiner::CLMiner(FarmFace& _farm, unsigned _index): m_useAsmKernel(s_asmVersion >= 1),
	Miner("cl-", _farm, _index){ }

CLMiner::~CLMiner()
{
	pause();
}

void CLMiner::detectAmdGpu(string gpu_name, unsigned char **_buffer, unsigned int *_bufferSize, unsigned int version)
{
	boost::algorithm::to_lower(gpu_name);
	m_useAsmKernel = true;

	if (gpu_name.find("hawaii") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_hawaii_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_hawaii_kernel);
	}
	else if (gpu_name.find("fiji") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_fiji_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_fiji_kernel);
	}
	else if (gpu_name.find("tahiti") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_tahiti_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_tahiti_kernel);
	}
	else if (gpu_name.find("pitcairn") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_pitcarin_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_pitcarin_kernel);
	}
	else if (gpu_name.find("tonga") != std::string::npos) {
		if (version <= 1) {
			if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_tonga_kernel1;
			if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_tonga_kernel1);
		}
		else {
			if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_tonga_kernel2;
			if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_tonga_kernel2);
		}
	}
	else if (gpu_name.find("ellesmere") != std::string::npos) {
		if (version <= 1) {
			if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_ellesmere_kernel1;
			if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_ellesmere_kernel1);
		}
		else {
			if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_ellesmere_kernel2;
			if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_ellesmere_kernel2);
		}
	}
	else if (gpu_name.find("baffin") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_baffin_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_baffin_kernel);
	}
	else if (gpu_name.find("gfx90") != std::string::npos) {
		if (_buffer != nullptr) *_buffer = (unsigned char *)CLASM_vega_kernel;
		if (_bufferSize != nullptr) *_bufferSize = sizeof(CLASM_vega_kernel);
	}
	else {
		m_useAsmKernel = false;
	}
}
void CLMiner::report(uint64_t _nonce, WorkPackage const& _w)
{
	assert(_nonce != 0);
	// TODO: Why re-evaluating?
	Result r = EthashAux::eval(_w.seed, _w.header, _nonce);
	if (r.value < _w.boundary) {
		farm.submitProof(Solution{ _nonce, r.mixHash, _w.header, _w.seed, _w.boundary });
	}
	else
		cwarn << "Invalid solution" << _nonce;
}

void CLMiner::kickOff()
{}

namespace
{
uint64_t randomNonce()
{
	static std::mt19937_64 s_gen(std::random_device{}());
	return std::uniform_int_distribution<uint64_t>{}(s_gen);
}
}

void CLMiner::workLoop()
{
	// Memory for zero-ing buffers. Cannot be static because crashes on macOS.
	uint32_t const c_zero = 0;

	uint64_t startNonce = 0;

	// The work package currently processed by GPU.
	WorkPackage current;
	current.header = h256{1u};
	current.seed = h256{1u};

	try {
		while (true)
		{
			const WorkPackage w = work();

			if (current.header != w.header)
			{
				// New work received. Update GPU data.
				auto localSwitchStart = std::chrono::high_resolution_clock::now();

				if (!w)
				{
					cllog << "No work. Pause for 3 s.";
					std::this_thread::sleep_for(std::chrono::seconds(3));
					continue;
				}

				cllog << "New work: header" << w.header << "target" << w.boundary.hex();

				if (current.seed != w.seed)
				{
					if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
					{
						while (s_dagLoadIndex < index)
							this_thread::sleep_for(chrono::seconds(1));
						++s_dagLoadIndex;
					}

					cllog << "New seed" << w.seed;
					init(w.seed);
				}

				// Upper 64 bits of the boundary.
				const uint64_t target = (uint64_t)(u64)((u256)w.boundary >> 192);
				assert(target > 0);

				// Update header constant buffer.
				m_queue.enqueueWriteBuffer(m_header, CL_FALSE, 0, w.header.size, w.header.data());
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_FALSE, 0, sizeof(c_zero), &c_zero);

				if (m_useAsmKernel) {
					m_asmSearchKernel.setArg(4, target);
					m_asmSearchKernel.setArg(0, m_searchBuffer);  // Supply output buffer to kernel.
				}
				else {
					m_searchKernel.setArg(4, target);
					m_searchKernel.setArg(0, m_searchBuffer);  // Supply output buffer to kernel.
				}



				// FIXME: This logic should be move out of here.
				if (w.exSizeBits >= 0)
					startNonce = w.startNonce | ((uint64_t)index << (64 - 4 - w.exSizeBits)); // This can support up to 16 devices.
				else
					startNonce = randomNonce();

				current = w;
				auto switchEnd = std::chrono::high_resolution_clock::now();
				auto globalSwitchTime = std::chrono::duration_cast<std::chrono::milliseconds>(switchEnd - workSwitchStart).count();
				auto localSwitchTime = std::chrono::duration_cast<std::chrono::microseconds>(switchEnd - localSwitchStart).count();
				cllog << "Switch time" << globalSwitchTime << "ms /" << localSwitchTime << "us";
			}

			// Read results.
			// TODO: could use pinned host pointer instead.
			uint32_t results[c_maxSearchResults + 1];
			m_queue.enqueueReadBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(results), &results);

			uint64_t nonce = 0;
			if ((results[0] & 0xF) > 0)
			{
				// Ignore results except the first one.
				nonce = startNonce + results[1];

				// Reset search buffer if any solution found.
				m_queue.enqueueWriteBuffer(m_searchBuffer, CL_TRUE, 0, sizeof(c_zero), &c_zero);
			}

			// Increase start nonce for following kernel execution.
			startNonce += m_globalWorkSize;

			// Run the kernel.
			if (m_useAsmKernel) {
				m_asmSearchKernel.setArg(3, startNonce);
				m_queue.enqueueNDRangeKernel(m_asmSearchKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);
			}
			else {
				m_searchKernel.setArg(3, startNonce);
				m_queue.enqueueNDRangeKernel(m_searchKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);
			}

			// Report results while the kernel is running.
			// It takes some time because ethash must be re-evaluated on CPU.
			if (nonce != 0)
				report(nonce, current);

			// Report hash count
			addHashCount(m_globalWorkSize);

			// Check if we should stop.
			if (shouldStop())
			{
				// Make sure the last buffer write has finished --
				// it reads local variable.
				m_queue.finish();
				break;
			}
		}
	}
	catch (cl::Error const& _e)
	{
		cwarn << "OpenCL Error:" << _e.what() << _e.err();
	}
}

void CLMiner::pause()
{}

unsigned CLMiner::getNumDevices()
{
	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return 0;

	vector<cl::Device> devices = getDevices(platforms, s_platformId);
	if (devices.empty())
	{
		cwarn << "No OpenCL devices found.";
		return 0;
	}
	return devices.size();
}

void CLMiner::listDevices()
{
	string outString ="\nListing OpenCL devices.\nFORMAT: [deviceID] deviceName\n";
	unsigned int i = 0;

	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return;
	for (unsigned j = 0; j < platforms.size(); ++j)
	{
		vector<cl::Device> devices = getDevices(platforms, j);
		for (auto const& device: devices)
		{
			outString += "[" + to_string(i) + "] " + device.getInfo<CL_DEVICE_NAME>() + "\n";
			outString += "\tCL_DEVICE_TYPE: ";
			switch (device.getInfo<CL_DEVICE_TYPE>())
			{
			case CL_DEVICE_TYPE_CPU:
				outString += "CPU\n";
				break;
			case CL_DEVICE_TYPE_GPU:
				outString += "GPU\n";
				break;
			case CL_DEVICE_TYPE_ACCELERATOR:
				outString += "ACCELERATOR\n";
				break;
			default:
				outString += "DEFAULT\n";
				break;
			}
			outString += "\tCL_DEVICE_GLOBAL_MEM_SIZE: " + to_string(device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_MEM_ALLOC_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()) + "\n";
			outString += "\tCL_DEVICE_MAX_WORK_GROUP_SIZE: " + to_string(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()) + "\n";
			++i;
		}
	}
	std::cout << outString;
}

bool CLMiner::configureGPU(
	unsigned _localWorkSize,
	unsigned _globalWorkSizeMultiplier,
	unsigned _platformId,
	uint64_t _currentBlock,
	unsigned _dagLoadMode,
	unsigned _dagCreateDevice,
	unsigned _version,
	unsigned _ethIntensity
)
{
	s_dagLoadMode = _dagLoadMode;
	s_dagCreateDevice = _dagCreateDevice;

	s_platformId = _platformId;

	_localWorkSize = ((_localWorkSize + 7) / 8) * 8;
	s_workgroupSize = _localWorkSize;
	s_initialGlobalWorkSize = _globalWorkSizeMultiplier * _localWorkSize;
	s_ethIntensity = _ethIntensity;

	if (_version >= 1) s_asmVersion = _version;
	uint64_t dagSize = ethash_get_datasize(_currentBlock);

	vector<cl::Platform> platforms = getPlatforms();
	if (platforms.empty())
		return false;
	if (_platformId >= platforms.size())
		return false;

	vector<cl::Device> devices = getDevices(platforms, _platformId);
	for (auto const& device: devices)
	{
		cl_ulong result = 0;
		device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
		if (result >= dagSize)
		{
			cnote <<
				"Found suitable OpenCL device [" << device.getInfo<CL_DEVICE_NAME>()
												 << "] with " << result << " bytes of GPU memory";
			return true;
		}

		cnote <<
			"OpenCL device " << device.getInfo<CL_DEVICE_NAME>()
							 << " has insufficient GPU memory." << result <<
							 " bytes of memory found < " << dagSize << " bytes of memory required";
	}

	cout << "No GPU device with sufficient memory was found. Can't GPU mine. Remove the -G argument" << endl;
	return false;
}


bool CLMiner::init(const h256& seed)
{
	EthashAux::LightType light = EthashAux::light(seed);

	// get all platforms
	try
	{
		vector<cl::Platform> platforms = getPlatforms();
		if (platforms.empty())
			return false;

		// use selected platform
		unsigned platformIdx = min<unsigned>(s_platformId, platforms.size() - 1);

		string platformName = platforms[platformIdx].getInfo<CL_PLATFORM_NAME>();
		ETHCL_LOG("Platform: " << platformName);

		int platformId = OPENCL_PLATFORM_UNKNOWN;
		if (platformName == "NVIDIA CUDA")
		{
			platformId = OPENCL_PLATFORM_NVIDIA;
		}
		else if (platformName == "AMD Accelerated Parallel Processing")
		{
			platformId = OPENCL_PLATFORM_AMD;
		}
		else if (platformName == "Clover")
		{
			platformId = OPENCL_PLATFORM_CLOVER;
		}

		// get GPU device of the default platform
		vector<cl::Device> devices = getDevices(platforms, platformIdx);
		if (devices.empty())
		{
			ETHCL_LOG("No OpenCL devices found.");
			return false;
		}

		// use selected device
		unsigned deviceId = s_devices[index] > -1 ? s_devices[index] : index;
		cl::Device& device = devices[min<unsigned>(deviceId, devices.size() - 1)];
		string device_version = device.getInfo<CL_DEVICE_VERSION>();
		ETHCL_LOG("Device:   " << device.getInfo<CL_DEVICE_NAME>() << " / " << device_version);

		string clVer = device_version.substr(7, 3);
		if (clVer == "1.0" || clVer == "1.1")
		{
			if (platformId == OPENCL_PLATFORM_CLOVER)
			{
				ETHCL_LOG("OpenCL " << clVer << " not supported, but platform Clover might work nevertheless. USE AT OWN RISK!");
			}
			else
			{
				ETHCL_LOG("OpenCL " << clVer << " not supported - minimum required version is 1.2");
				return false;
			}
		}

		char options[256];
		int computeCapability = 0;
		if (platformId == OPENCL_PLATFORM_NVIDIA) {
			cl_uint computeCapabilityMajor;
			cl_uint computeCapabilityMinor;
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), &computeCapabilityMajor, NULL);
			clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), &computeCapabilityMinor, NULL);

			computeCapability = computeCapabilityMajor * 10 + computeCapabilityMinor;
			int maxregs = computeCapability >= 35 ? 72 : 63;
			sprintf(options, "-cl-nv-maxrregcount=%d", maxregs);
		}
		else {
			sprintf(options, "%s", "");
		}

		cllog << "Creating context...";
		// create context
		m_context = cl::Context(vector<cl::Device>(&device, &device + 1));
		m_queue = cl::CommandQueue(m_context, device);

		// make sure that global work size is evenly divisible by the local workgroup size
		m_workgroupSize = s_workgroupSize;
		m_globalWorkSize = s_initialGlobalWorkSize;
		if (m_globalWorkSize % m_workgroupSize != 0)
			m_globalWorkSize = ((m_globalWorkSize / m_workgroupSize) + 1) * m_workgroupSize;

		m_dagSize = ethash_get_datasize(light->light->block_number);
		uint32_t dagSize128 = (unsigned)(m_dagSize / ETHASH_MIX_BYTES);
		uint32_t lightSize64 = (unsigned)(light->data().size() / sizeof(node));

		// patch source code
		// note: CLMiner_kernel is simply ethash_cl_miner_kernel.cl compiled
		// into a byte array by bin2h.cmake. There is no need to load the file by hand in runtime
		// TODO: Just use C++ raw string literal.

		string code(CLMiner_kernel, CLMiner_kernel + sizeof(CLMiner_kernel));
		addDefinition(code, "GROUP_SIZE", m_workgroupSize);
		addDefinition(code, "DAG_SIZE", dagSize128);
		addDefinition(code, "LIGHT_SIZE", lightSize64);
		addDefinition(code, "ACCESSES", ETHASH_ACCESSES);
		addDefinition(code, "MAX_OUTPUTS", c_maxSearchResults);
		addDefinition(code, "PLATFORM", platformId);
		addDefinition(code, "COMPUTE", computeCapability);

		// create miner OpenCL program
		
		cl::Program::Sources sources{{code.data(), code.size()}};
		cl::Program program(m_context, sources);
		try
		{
			program.build({device}, NULL);
			cllog << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
		}
		catch (cl::Error const&)
		{
			cwarn << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			return false;
		}

		string gpu_name = device.getInfo<CL_DEVICE_NAME>();
		unsigned char *binary_ptr = nullptr;
		unsigned int binary_size = 0;
	
		unsigned int computeUnits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

		if(m_useAsmKernel)
			this->detectAmdGpu(gpu_name, &binary_ptr, &binary_size, s_asmVersion);

		cl::Program::Binaries binaries{ { binary_ptr, binary_size} };
		cl::Program asmProgram; 
		if (m_useAsmKernel) {
			cl::Program program(m_context, { device }, binaries);
			cllog << "Using optimized kernel";
			try
			{
				program.build({ device }, options);
				cllog << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			}
			catch (cl::Error const&)
			{
				cwarn << "Build info:" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
				return false;
			}

			asmProgram = program;
			m_workgroupSize = 64;

			computeUnits = computeUnits == 14 ? 36 : computeUnits;
			m_globalWorkSize = (computeUnits << 14)*s_ethIntensity;
		}

		// create buffer for dag
		try
		{
			cllog << "Creating buffers";
			m_light = cl::Buffer(m_context, CL_MEM_READ_ONLY, light->data().size());
			m_dag = cl::Buffer(m_context, CL_MEM_READ_ONLY, m_dagSize);

			cllog << "Loading kernels";
			m_searchKernel = cl::Kernel(program, "ethash_search");
			m_dagKernel = cl::Kernel(program, "ethash_calculate_dag_item");

			if (m_useAsmKernel) {
				m_asmSearchKernel = cl::Kernel(asmProgram, "k_r1a");
				m_asmDagKernel = cl::Kernel(asmProgram, "kd");
			}
			
			cllog << "Writing light cache buffer";
			m_queue.enqueueWriteBuffer(m_light, CL_TRUE, 0, light->data().size(), light->data().data());
		}
		catch (cl::Error const& err)
		{
			cwarn << "Creating DAG buffer failed:" << err.what() << err.err();
			return false;
		}
		// create buffer for header
		ETHCL_LOG("Creating buffer for header.");

		// load the lut for the ASM kernels, the SPIR one doesn't need this
		// but the asm ones do. SPIR kernels just ignore it
		assert(sizeof(Keccak_rho_etc_LUT) > 32);
		m_header = cl::Buffer(m_context, CL_MEM_READ_ONLY, sizeof(Keccak_rho_etc_LUT));
		m_queue.enqueueWriteBuffer(m_header, CL_TRUE, 0, sizeof(Keccak_rho_etc_LUT), Keccak_rho_etc_LUT);

		m_searchKernel.setArg(1, m_header);
		m_searchKernel.setArg(2, m_dag);
		m_searchKernel.setArg(5, ~0u);  // Pass this to stop the compiler unrolling the loops.

		if (m_useAsmKernel) {
			m_asmSearchKernel.setArg(1, m_header);
			m_asmSearchKernel.setArg(2, m_dag);
			m_asmSearchKernel.setArg(5, ~0u);  // Pass this to stop the compiler unrolling the loops.

			m_asmSearchKernel.setArg(6, m_dagSize >> 7);  // Supply output buffer to kernel.
			m_asmSearchKernel.setArg(7, exponent_table[light->light->block_number / ETHASH_EPOCH_LENGTH]);
			m_asmSearchKernel.setArg(8, factor_table[light->light->block_number / ETHASH_EPOCH_LENGTH]);

			m_asmSearchKernel.setArg(9, m_header);
			m_asmSearchKernel.setArg(10, 0);
			m_asmSearchKernel.setArg(11, 0);
		}

		// create mining buffers
		ETHCL_LOG("Creating mining buffer");
		m_searchBuffer = cl::Buffer(m_context, CL_MEM_READ_WRITE, 1024);
		m_queue.enqueueFillBuffer(m_searchBuffer, (unsigned char)0, 0, 1024);

		
		if (m_useAsmKernel) {
			cllog << "Generating fast DAG";
			m_asmDagKernel.setArg(0, m_dag);
			m_asmDagKernel.setArg(1, m_light);
			m_asmDagKernel.setArg(2, (unsigned int)m_dagSize);
			m_asmDagKernel.setArg(3, (unsigned int)light->data().size());

			uint32_t const work = (uint32_t)(m_dagSize / sizeof(node));
			int global_worksize = (work + (2048 - work % 2048)) / 16;

			for (int i = 0; i < 16; ++i) {
				m_asmDagKernel.setArg(4, (unsigned int)(i * global_worksize));

				m_queue.enqueueNDRangeKernel(
					m_asmDagKernel,
					cl::NullRange,
					global_worksize,
					128
				);
				m_queue.finish();
				cllog << "DAG" << int(100.f*i / 15.f) << "%...";
			}
		}
		else {

			cllog << "Generating DAG";
			uint32_t const work = (uint32_t)(m_dagSize / sizeof(node));
			uint32_t fullRuns = work / m_globalWorkSize;
			uint32_t const restWork = work % m_globalWorkSize;
			if (restWork > 0) fullRuns++;

			m_dagKernel.setArg(1, m_light);
			m_dagKernel.setArg(2, m_dag);
			m_dagKernel.setArg(3, ~0u);

			for (uint32_t i = 0; i < fullRuns; i++)
			{
				m_dagKernel.setArg(0, i * m_globalWorkSize);
				m_queue.enqueueNDRangeKernel(m_dagKernel, cl::NullRange, m_globalWorkSize, m_workgroupSize);
				m_queue.finish();
				cllog << "DAG" << int(100.0f * i / fullRuns) << '%';
			}
		}
	}
	catch (cl::Error const& err)
	{
		cwarn << err.what() << "(" << err.err() << ")";
		return false;
	}
	return true;
}
