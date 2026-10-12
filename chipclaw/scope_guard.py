"""Conservative lexical policy for explicit requests for missing instance values."""
import re

def missing_instance_value(question):
    personal=bool(re.search(r'我(?:的|手中|这一|这块|这颗|当前)|这(?:块|颗)|my\b|this\s+(?:device|board|chip)',question,re.I))
    unavailable=bool(re.search(r'没有|未提供|不知道|未知|without|not\s+(?:provided|given)|unknown',question,re.I))
    value_request=bool(re.search(r'给出|告诉|是多少|what\s+is|give|provide|tell',question,re.I))
    instance_target=bool(re.search(r'实测|实际(?:电流|功率|阻值|温度|值|出厂)|精确|确定(?:数值|值|十六进制)|完整.*(?:ID|数字)|'
                                  r'exact|actual|measured|unique\s+id',question,re.I))
    # An explicit request for methods/ranges/default specifications remains answerable.
    generic=bool(re.search(r'如何(?:测量|读取|计算)|测量方法|读取方法|what.*(?:procedure|method)|how\s+(?:to|can|do)',question,re.I))
    refuse=personal and unavailable and value_request and instance_target and not generic
    return {'refuse':refuse,'rule':'explicit_missing_instance_value' if refuse else None,
            'reason':'缺少该设备的测量、配置或读取数据，型号手册不能确定所请求的个体值。' if refuse else '',
            'note':'Lexical fail-closed policy for explicit missing-input requests; not a general unknown-question classifier.'}

