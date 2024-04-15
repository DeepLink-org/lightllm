#处理sot的逻辑函数
    
def format_outline_prompt(outline_prompt, request):
        splits = outline_prompt.split("[ROLESWITCHING assistant:]")
        #print(splits[1])
        if len(splits) == 1:
            return splits[0].format(request=request), None
        return splits[0].format(request=request), splits[1].format(request=request)


def format_point_prompt(point_prompt, request, outline, point, point_outline):
    splits = point_prompt.split("[ROLESWITCHING assistant:]")
    if len(splits) == 1:
        return (
            splits[0].format(
                request=request,
                outline=outline,
                point=point,
                point_outline=point_outline,
            ),
            None,
        )
    return [
        split.format(
            request=request,
            outline=outline,
            point=point,
            point_outline=point_outline,
        )
        for split in splits
    ]