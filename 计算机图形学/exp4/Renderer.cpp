//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include <thread>
#include <atomic>
#include "Scene.hpp"
#include "Renderer.hpp"


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.00001;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    const uint32_t width = scene.width;
    const uint32_t height = scene.height;
    std::vector<Vector3f> framebuffer(width * height, Vector3f(0.0f));

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = width / (float)height;
    Vector3f eye_pos(278, 273, -800);

    // change the spp value to change sample ammount
    int spp = 64;
    std::cout << "SPP: " << spp << "\n";

    unsigned int threadCount = std::max(1u, std::thread::hardware_concurrency());
    unsigned int blockSize = std::max(
        1u, (height + threadCount - 1) / threadCount);
    std::atomic<uint32_t> rowsFinished{0};

    auto renderRows = [&](uint32_t startY, uint32_t endY) {
        for (uint32_t j = startY; j < endY && j < height; ++j) {
            for (uint32_t i = 0; i < width; ++i) {
                Vector3f pixelColor(0.0f);
                float x = (2 * (i + 0.5f) / (float)width - 1) *
                          imageAspectRatio * scale;
                float y = (1 - 2 * (j + 0.5f) / (float)height) * scale;
                Vector3f dir = normalize(Vector3f(-x, y, 1));
                for (int k = 0; k < spp; k++) {
                    pixelColor += scene.castRay(Ray(eye_pos, dir), 0);
                }
                framebuffer[j * width + i] = pixelColor / (float)spp;
            }
            uint32_t finished = rowsFinished.fetch_add(1) + 1;
            if (finished % 10 == 0 || finished == height) {
                UpdateProgress(finished / (float)height);
            }
        }
    };

    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < threadCount; ++t) {
        uint32_t startY = t * blockSize;
        uint32_t endY = (t + 1) * blockSize;
        if (startY >= height)
            break;
        workers.emplace_back(renderRows, startY, endY);
    }
    for (auto& worker : workers) {
        if (worker.joinable())
            worker.join();
    }
    UpdateProgress(1.f);

    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
