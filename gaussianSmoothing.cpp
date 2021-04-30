#include <Kokkos_Core.hpp>
#include <fstream>
#include <chrono>
#include <cstdio>
#include < string>

using std::string;
using std::ios;
using std::ifstream;
using std::ofstream;

double duration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end )
{
	return std::chrono::duration<double< std::milli>(end - start).count();
}

int main(int argc, char ** argv) {
	
	Kokkos::initialize(argc, argv);
	{
		string filename = "HK-7_left_H6D-400c-MS.bmp";
		ifstream fin(filename, ios::in | ios::binary);
		
		if (!fin.is_open()) {
			printf("File not opened\n");
			return -1;
		}
		// The first 14 bytes are the header, containing 4 values. Get those 4 values.
		char header[2];
		uint32_t filesize;
		uint32_t dummy;
		uint32_t offset;
		fin.read(header, 2);
		fin.read((char*)&filesize, 4);
		fin.read((char*)&dummy, 4);
		fin.read((char*)&offset, 4);
		printf("header: %c%c\n", header[0], header[1]);
		printf("filesize: %u\n", filesize);
		printf("dummy %u\n", dummy);
		printf("offset: %u\n", offset);
		int32_t sizeOfHeader;
		int32_t width;
		int32_t height;
		fin.read((char*)&sizeOfHeader, 4);
		fin.read((char*)&width, 4);
		fin.read((char*)&height, 4);
		printf("The width: %d\n", width);
		printf("The height: %d\n", height);
		uint16_t numColorPanes;
		uint16_t numBitsPerPixel;
		fin.read((char*)&numColorPanes, 2);
		fin.read((char*)&numBitsPerPixel, 2);
		printf("The number of bits per pixel: %u\n", numBitsPerPixel);
		if (numBitsPerPixel == 24) {
			printf("This bitmap uses rgb, where the first byte is blue, the second byte is greent, third byte is red.\n");
		}

		// Jump to the offset where the bitmap pixel data starts
		fin.seekg(offset, ios::beg);

		// Read the data part of the file
		unsigned char* h_buffer = new unsigned char[filesize - offset];
		fin.read((char*)h_buffer, filesize - offset);
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point end;
		printf("The first pixel is located in the bottom left.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[0], h_buffer[1], h_buffer[2]);
		printf("The first pixel is located in the bottom left.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[3], h_buffer[4], h_buffer[5]);

		// TODO: Read the image into Kokkos views --> see lecture
		printf("Kokkos execution space is %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
		int rows{23200};
		int cols{17400};
		//Kokkos::View<float**, Kokkos::LayoutRight> portableInput("portableInput", rows, cols);
		//Kokkos::View<float**, Kokkos::LayoutRight> portableOutput("portableOutput", rows, cols);
		//Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostInput = create_mirror(portableInput);
		//Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostOutput = create_mirror(portableOutput);

		// Prof said height is # rows, and width is # cols
		// Device side
		Kokkos::View<float**, Kokkos::LayoutRight> blueInput("blueInput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight> greenInput("greenInput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight> redInput("redInput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostBlueInput = create_mirror(blueInput);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostGreenInput = create_mirror(greenInput);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostRedInput = create_mirror(redInput);

		// Host side
		Kokkos::View<float**, Kokkos::LayoutRight> blueOutput("blueOutput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight> greenOutput("greenOutput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight> redOutput("redOutput", height, width);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostBlueOutput = create_mirror(blueOutput);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostGreenOutput = create_mirror(greenOutput);
		Kokkos::View<float**, Kokkos::LayoutRight>::HostMirror hostRedOutput = create_mirror(redOutput);

		//for (int i = 0; i < rows; i++) {
		//	for (int j = 0; j < cols; j+=3) {
		//		blueInput(i, j) = i + j;
		//		greenInput(i, j) = i + j + 1;
		//		redInput(i, j) = i + j + 2;
		//	}
		//}
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				blueInput(i, j) = i + j;
				greenInput(i, j) = i + j;
				redInput(i, j) = i + j;
			}
		}


		printf("h_buffer[10,10] is: %d, kokkos[10,10] is %f\n", h_buffer[20], hostBlueInput(10, 10));
		printf("h_buffer[10,10] is: %d, kokkos[10,10] is %f\n", h_buffer[21], hostGreenInput(10, 10));
		printf("h_buffer[10,10] is: %d, kokkos[10,10] is %f\n", h_buffer[22], hostRedInput(10, 10));

		//Kokkos::deep_copy(blueInput, greenInput, redInput, blueOutput, hostBlueInput, hostGreenInput, hostRedInput);

		// Kokkos parallel stuff...
		//Kokkos::parallel_for(Kokkos::RangePolicy<>(0, rows*cols), KOKKOS_LAMBDA const int n) {
		//	int i = n / rows;
		//	int j = n % cols;
		//	blueOutput(i, j) = blueInput(i, j);
		//	greenOutput(i, j) = greenOutput(i, j);
		//	redIOutput(i, j) = redOutput(i, j);

		//});

		//Kokkos::deep_copy(blueOutput, greenOutput, redOutput, hostBlueOutput, hostGreenOutput, hostRedOutput);

		delete[] h_buffer;



		start = std::chrono::high_resolution_clock::now();
		// TODO: Perform the blurring
		end = std::chrono::high_resolution_clock::now();
		printf("Time - %g ms\n", duration(start, end));

		// TODO: Verification
		printf("The red, green, blue at (8353, 9111) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (8351, 9113) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (10559, 10611) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (10818, 20226) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);

		//Print out to file output.bmp
		string outputFile = "output.bmp";
		ofstream fout;
		fout.open(outputFile, ios::binary);

		// Copy of the old headers into the new output
		fin.seekg(0, ios::beg);
		// Read the data part of the file
		char* headers = new char[offset];
		fin.read(headers, offset);
		fout.seekp(0, ios::beg);
		fout.write(headers, offset);
		delete[] headers;

		fout.seekp(offset, ios::beg);
		// TODO: Copy out the rest of the view to file (hint, use fout.put())
		fout.close();
	}
	Kokkos::finalize

}