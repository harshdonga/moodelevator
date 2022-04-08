from django.http import JsonResponse
class CustomMiddleware:
    ALLOWED_IPs = ["192.168.30.67","10.6.33.141"]
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        ip_address = self.getIPAddress(request)
        #allowed = ip_address in self.ALLOWED_IPs
        allowed = True
        if allowed:
            response = self.get_response(request)
        else:
            response = JsonResponse({"status":"Not Allowed"})
        return response

    def getIPAddress(self,request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip