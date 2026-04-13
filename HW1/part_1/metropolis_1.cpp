#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

constexpr int kDefaultSize = 256;

double WrapCoord(double x, int n) {
    const double m = std::fmod(x, static_cast<double>(n));
    return (m < 0.0) ? (m + n) : m;
}

int WrapIndex(double x, int n) {
    return static_cast<int>(std::floor(WrapCoord(x, n)));
}

double Luminance(const unsigned char* pixel, int channels) {
    if (channels <= 1) {
        return pixel[0] / 255.0;
    }
    const double r = pixel[0] / 255.0;
    const double g = pixel[1] / 255.0;
    const double b = pixel[2] / 255.0;
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

std::vector<double> ToGray(
    const unsigned char* data, int width, int height, int channels) {
    std::vector<double> gray(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned char* px = data + (y * width + x) * channels;
            gray[y * width + x] = Luminance(px, channels);
        }
    }
    return gray;
}

std::vector<double> ResizeBilinear(
    const std::vector<double>& src, int srcW, int srcH, int dstW, int dstH) {
    std::vector<double> dst(dstW * dstH);
    for (int y = 0; y < dstH; ++y) {
        const double gy = (static_cast<double>(y) + 0.5) * srcH / dstH - 0.5;
        const int y0 = std::clamp(static_cast<int>(std::floor(gy)), 0, srcH - 1);
        const int y1 = std::clamp(y0 + 1, 0, srcH - 1);
        const double ty = gy - std::floor(gy);

        for (int x = 0; x < dstW; ++x) {
            const double gx = (static_cast<double>(x) + 0.5) * srcW / dstW - 0.5;
            const int x0 = std::clamp(static_cast<int>(std::floor(gx)), 0, srcW - 1);
            const int x1 = std::clamp(x0 + 1, 0, srcW - 1);
            const double tx = gx - std::floor(gx);

            const double c00 = src[y0 * srcW + x0];
            const double c10 = src[y0 * srcW + x1];
            const double c01 = src[y1 * srcW + x0];
            const double c11 = src[y1 * srcW + x1];

            const double c0 = c00 * (1.0 - tx) + c10 * tx;
            const double c1 = c01 * (1.0 - tx) + c11 * tx;
            dst[y * dstW + x] = c0 * (1.0 - ty) + c1 * ty;
        }
    }
    return dst;
}

double Eval(const std::vector<double>& image, int w, int h, double x, double y) {
    const int xi = WrapIndex(x, w);
    const int yi = WrapIndex(y, h);
    constexpr double epsilon = 1e-6;
    return image[yi * w + xi] + epsilon;
}

std::vector<unsigned char> MetropolisImageCopy(
    const std::vector<double>& image,
    int w,
    int h,
    int samplesPerPixel,
    unsigned int seed) {
    const int totalSamples = std::max(1, samplesPerPixel) * w * h;
    const int burnIn = std::max(200, totalSamples / 100);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::uniform_real_distribution<double> uniX(0.0, static_cast<double>(w));
    std::uniform_real_distribution<double> uniY(0.0, static_cast<double>(h));
    std::normal_distribution<double> normal(0.0, 8.0);

    double cx = uniX(rng);
    double cy = uniY(rng);
    double cf = Eval(image, w, h, cx, cy);

    std::vector<double> hits(w * h, 0.0);
    int accepted = 0;

    for (int i = 0; i < burnIn + totalSamples; ++i) {
        double nx = cx;
        double ny = cy;

        if (uni01(rng) < 0.1) {
            nx = uniX(rng);
            ny = uniY(rng);
        } else {
            nx += normal(rng);
            ny += normal(rng);
        }

        // Use toroidal wrapping instead of clamping to preserve proposal symmetry.
        nx = WrapCoord(nx, w);
        ny = WrapCoord(ny, h);

        const double nf = Eval(image, w, h, nx, ny);
        const double accept = std::min(1.0, nf / cf);
        if (uni01(rng) < accept) {
            cx = nx;
            cy = ny;
            cf = nf;
            ++accepted;
        }

        if (i >= burnIn) {
            const int px = WrapIndex(cx, w);
            const int py = WrapIndex(cy, h);
            hits[py * w + px] += 1.0;
        }
    }

    double sumF = 0.0;
    for (double v : image) {
        sumF += v;
    }
    if (sumF <= 0.0) {
        sumF = 1.0;
    }

    std::vector<unsigned char> out(w * h);
    for (int i = 0; i < w * h; ++i) {
        // If samples follow p(x)=f(x)/sum(f), then f(x) ~= p(x)*sum(f).
        const double p = hits[i] / static_cast<double>(totalSamples);
        const double reconstructed = std::clamp(p * sumF, 0.0, 1.0);
        out[i] = static_cast<unsigned char>(std::round(reconstructed * 255.0));
    }

    std::cout << "Total samples: " << totalSamples << "\n";
    std::cout << "Accepted moves: " << accepted << "\n";
    std::cout << "Acceptance rate: " << (100.0 * accepted / (burnIn + totalSamples))
              << "%\n";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input-image> <output-image.png> [samples-per-pixel] [seed]\n";
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    const int spp = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 8;
    const unsigned int seed = (argc >= 5) ? static_cast<unsigned int>(std::stoul(argv[4]))
                                           : 42u;

    int w = 0, h = 0, channels = 0;
    unsigned char* raw = stbi_load(inputPath.c_str(), &w, &h, &channels, 0);
    if (!raw) {
        std::cerr << "Failed to load input image: " << inputPath << "\n";
        return 1;
    }

    std::vector<double> gray = ToGray(raw, w, h, channels);
    stbi_image_free(raw);

    if (w != kDefaultSize || h != kDefaultSize) {
        std::cout << "Input image is " << w << "x" << h
                  << ", resizing to 256x256.\n";
        gray = ResizeBilinear(gray, w, h, kDefaultSize, kDefaultSize);
        w = kDefaultSize;
        h = kDefaultSize;
    }

    std::vector<unsigned char> copied = MetropolisImageCopy(gray, w, h, spp, seed);

    if (!stbi_write_png(outputPath.c_str(), w, h, 1, copied.data(), w)) {
        std::cerr << "Failed to write output image: " << outputPath << "\n";
        return 1;
    }

    std::cout << "Done. Wrote output to: " << outputPath << "\n";
    return 0;
}