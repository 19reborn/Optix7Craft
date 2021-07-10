
#ifndef OPTIX_SAMPLES_COLLIDEBOX_H
#define OPTIX_SAMPLES_COLLIDEBOX_H

#include <vector_types.h>
#include <optix_types.h>
#include <sutil/vec_math.h>

class CollideBox {
public:
    float3 center;
    float3 size;

    CollideBox(float3 c, float3 s): center(c), size(s) {}

    static bool collide_check(const CollideBox& A, const CollideBox& B) {
        // 用法：CollideBox::collide_check(pModel->get_collideBox(), pEntity->get_collideBox())
        // 不用严格按照pModel、pEntity的顺序，只要确保是CollideBox就行
        // 另外，我们默认碰撞箱之间没有角度，也就是说都是这样的： | | | |

        if(fabs(A.center.y - B.center.y) >= A.size.y + B.size.y) {
            return false;
        }
        if(fabs(A.center.x - B.center.x) >= A.size.x + B.size.x) {
            return false;
        }
        if(fabs(A.center.z - B.center.z) >= A.size.z + B.size.z) {
            return false;
        }

        return true;

    }
    bool check_collide_at(float3 pos) const {
        // 检测是否对该点<pos>产生了碰撞
        // 表面也算碰撞
        if(fabs(pos.x - center.x) > size.x) return false;
        if(fabs(pos.y - center.y) > size.y) return false;
        if(fabs(pos.z - center.z) > size.z) return false;
        return true;
    }
};

#endif //OPTIX_SAMPLES_COLLIDEBOX_H
